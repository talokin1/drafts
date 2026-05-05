import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


# =========================
# CONFIG
# =========================

RANDOM_STATE = 42

TARGET_NAME = "ASSETS"      # якщо в тебе вже задано вище — можеш прибрати
SEGMENT_COL = "FIRM_TYPE"   # якщо в тебе вже задано вище — можеш прибрати

MIN_ACTIVE_TARGET = 50      # усе нижче вважаємо нулем
CAP_Q = 0.95                # сильний захист від завищення хвоста

MODEL_PATH = "assets_bucket_ev_model.pkl"


# =========================
# DATA PREP
# =========================

X_model = X.copy()

if not isinstance(y, pd.Series):
    y_model = pd.Series(y, index=X_model.index, name=TARGET_NAME)
else:
    y_model = y.copy()
    y_model.index = X_model.index

y_model = np.clip(y_model, 0, None)
y_model = pd.Series(np.where(y_model < MIN_ACTIVE_TARGET, 0, y_model), index=X_model.index)

segments = df.loc[X_model.index, SEGMENT_COL].copy()

work_df = X_model.copy()
work_df["_target"] = y_model
work_df["_segment"] = segments.astype(str)

features = list(X_model.columns)

cat_cols = [
    c for c in features
    if X_model[c].dtype.name in ("object", "category")
]

for c in cat_cols:
    work_df[c] = work_df[c].astype("category")

cat_values = {
    c: list(work_df[c].cat.categories)
    for c in cat_cols
}


def get_model_group(segment):
    if str(segment).upper() == "LARGE":
        return "LARGE"
    return "MICRO_SMALL"


work_df["_model_group"] = work_df["_segment"].apply(get_model_group)


# =========================
# HELPERS
# =========================

def make_bucket_edges(y_train, n_quantiles=7):
    """
    Робить bucket-и тільки по positive target.
    0 bucket окремо: target == 0.
    """
    pos = y_train[y_train >= MIN_ACTIVE_TARGET]

    if len(pos) < 30:
        return np.array([-np.inf, np.inf])

    qs = np.linspace(0, 1, n_quantiles + 1)
    edges = np.quantile(pos, qs)

    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)

    if len(edges) < 3:
        edges = np.array([-np.inf, np.inf])

    return edges


def assign_buckets(y, edges):
    """
    bucket 0 = zero clients.
    positive buckets = 1, 2, 3...
    """
    y = pd.Series(y).copy()
    bucket = pd.Series(0, index=y.index, dtype=int)

    mask_pos = y >= MIN_ACTIVE_TARGET

    if mask_pos.sum() > 0:
        positive_bucket = pd.cut(
            y.loc[mask_pos],
            bins=edges,
            labels=False,
            include_lowest=True
        ) + 1

        bucket.loc[mask_pos] = positive_bucket.astype(int)

    return bucket


def make_bucket_medians(y_train, bucket_train):
    medians = {}

    all_buckets = sorted(bucket_train.unique())

    for b in all_buckets:
        if b == 0:
            medians[int(b)] = 0.0
        else:
            vals = y_train[bucket_train == b]
            if len(vals) == 0:
                medians[int(b)] = 0.0
            else:
                medians[int(b)] = float(np.median(vals))

    return medians


def prepare_X_for_model(df_part, features, cat_cols, cat_values):
    X_part = df_part[features].copy()

    for c in cat_cols:
        X_part[c] = pd.Categorical(
            X_part[c],
            categories=cat_values[c]
        )

    return X_part


def predict_bucket_model(model_pack, X_new, segment_new):
    X_new = X_new.copy()
    segment_new = pd.Series(segment_new, index=X_new.index).astype(str)

    preds = pd.Series(0.0, index=X_new.index)

    for group_name, pack in model_pack["groups"].items():
        group_mask = segment_new.apply(get_model_group) == group_name

        if group_mask.sum() == 0:
            continue

        X_part = X_new.loc[group_mask, model_pack["features"]].copy()

        for c in model_pack["cat_cols"]:
            X_part[c] = pd.Categorical(
                X_part[c],
                categories=model_pack["cat_values"][c]
            )

        model = pack["model"]
        medians = pack["bucket_medians"]

        proba = model.predict_proba(X_part)
        classes = model.classes_

        pred_raw = np.zeros(len(X_part))

        for i, cls in enumerate(classes):
            pred_raw += proba[:, i] * medians.get(int(cls), 0.0)

        pred_raw = pd.Series(pred_raw, index=X_part.index)

        # segment-level calibration
        for seg in segment_new.loc[group_mask].unique():
            seg_mask = segment_new.loc[group_mask] == seg
            factor = pack["calibration_factors"].get(seg, 1.0)
            cap = pack["segment_caps"].get(seg, np.inf)

            pred_raw.loc[seg_mask] = pred_raw.loc[seg_mask] * factor
            pred_raw.loc[seg_mask] = np.clip(pred_raw.loc[seg_mask], 0, cap)

        preds.loc[group_mask] = pred_raw

    return preds.values


def calc_metrics(y_true, y_pred, segment):
    df_m = pd.DataFrame({
        "segment": segment.astype(str),
        "y_true": y_true,
        "y_pred": y_pred
    })

    df_m["abs_error"] = np.abs(df_m["y_true"] - df_m["y_pred"])

    rows = []

    for seg_name, part in [("УСІ ДАНІ (Overall)", df_m)] + list(df_m.groupby("segment", observed=True)):
        y_t = part["y_true"].values
        y_p = part["y_pred"].values

        true_sum = y_t.sum()
        pred_sum = y_p.sum()

        rows.append({
            "Segment": seg_name,
            "Samples_Count": len(part),
            "MAE": mean_absolute_error(y_t, y_p),
            "MedAE": median_absolute_error(y_t, y_p),
            "R2": r2_score(y_t, y_p),
            "True_Sum": true_sum,
            "Pred_Sum": pred_sum,
            "Sum_Ratio": pred_sum / true_sum if true_sum > 0 else np.nan,
            "True_Mean": np.mean(y_t),
            "Pred_Mean": np.mean(y_p),
            "Zero_True_Rate": np.mean(y_t == 0),
            "Zero_Pred_Rate": np.mean(y_p == 0),
        })

    return pd.DataFrame(rows)


# =========================
# TRAIN
# =========================

model_pack = {
    "features": features,
    "cat_cols": cat_cols,
    "cat_values": cat_values,
    "min_active_target": MIN_ACTIVE_TARGET,
    "cap_q": CAP_Q,
    "groups": {}
}

all_val_parts = []

for group_name in sorted(work_df["_model_group"].unique()):
    print("=" * 80)
    print(f"TRAIN GROUP: {group_name}")

    df_g = work_df[work_df["_model_group"] == group_name].copy()

    stratify_col = (df_g["_target"] >= MIN_ACTIVE_TARGET).astype(int)

    train_idx, temp_idx = train_test_split(
        df_g.index,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=stratify_col
    )

    temp_df = df_g.loc[temp_idx].copy()
    stratify_temp = (temp_df["_target"] >= MIN_ACTIVE_TARGET).astype(int)

    cal_idx, val_idx = train_test_split(
        temp_df.index,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=stratify_temp
    )

    train_df = df_g.loc[train_idx].copy()
    cal_df = df_g.loc[cal_idx].copy()
    val_df = df_g.loc[val_idx].copy()

    y_train = train_df["_target"]
    y_cal = cal_df["_target"]
    y_val = val_df["_target"]

    edges = make_bucket_edges(y_train)
    bucket_train = assign_buckets(y_train, edges)
    bucket_medians = make_bucket_medians(y_train, bucket_train)

    X_train = prepare_X_for_model(train_df, features, cat_cols, cat_values)
    X_cal = prepare_X_for_model(cal_df, features, cat_cols, cat_values)
    X_val = prepare_X_for_model(val_df, features, cat_cols, cat_values)

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=8.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    clf.fit(
        X_train,
        bucket_train,
        eval_set=[(X_cal, assign_buckets(y_cal, edges))],
        eval_metric="multi_logloss",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    temp_pack = {
        "model": clf,
        "edges": edges,
        "bucket_medians": bucket_medians,
        "calibration_factors": {},
        "segment_caps": {}
    }

    # calibration на calibration split
    cal_pred_raw = np.zeros(len(X_cal))
    cal_proba = clf.predict_proba(X_cal)

    for i, cls in enumerate(clf.classes_):
        cal_pred_raw += cal_proba[:, i] * bucket_medians.get(int(cls), 0.0)

    cal_pred_raw = pd.Series(cal_pred_raw, index=cal_df.index)

    # calibration factor + cap по оригінальних сегментах
    for seg in cal_df["_segment"].unique():
        seg_mask = cal_df["_segment"] == seg

        true_sum = y_cal.loc[seg_mask].sum()
        pred_sum = cal_pred_raw.loc[seg_mask].sum()

        if pred_sum > 0 and true_sum > 0:
            factor = true_sum / pred_sum
        else:
            factor = 1.0

        factor = float(np.clip(factor, 0.05, 2.0))

        pos_train_seg = train_df.loc[
            (train_df["_segment"] == seg) &
            (train_df["_target"] >= MIN_ACTIVE_TARGET),
            "_target"
        ]

        if len(pos_train_seg) > 20:
            cap = float(np.quantile(pos_train_seg, CAP_Q))
        else:
            cap = float(np.quantile(train_df["_target"], CAP_Q))

        temp_pack["calibration_factors"][seg] = factor
        temp_pack["segment_caps"][seg] = cap

    model_pack["groups"][group_name] = temp_pack

    # validation prediction
    val_pred = predict_bucket_model(
        model_pack={
            **model_pack,
            "groups": {group_name: temp_pack}
        },
        X_new=X_val,
        segment_new=val_df["_segment"]
    )

    val_part = pd.DataFrame({
        "segment": val_df["_segment"].values,
        "y_true": y_val.values,
        "y_pred": val_pred
    }, index=val_df.index)

    all_val_parts.append(val_part)

    print(calc_metrics(y_val.values, val_pred, val_df["_segment"]).round(4))


# =========================
# FINAL VALIDATION RESULT
# =========================

validation_df = pd.concat(all_val_parts).sort_index()

validation_result = calc_metrics(
    validation_df["y_true"].values,
    validation_df["y_pred"].values,
    validation_df["segment"]
)

display(validation_result.round(4))


# =========================
# SAVE MODEL
# =========================

joblib.dump(model_pack, MODEL_PATH)
print(f"Saved to: {MODEL_PATH}")


# =========================
# INFERENCE EXAMPLE
# =========================

loaded_model = joblib.load(MODEL_PATH)

# Для інференсу на тих самих X:
assets_pred = predict_bucket_model(
    model_pack=loaded_model,
    X_new=X,
    segment_new=df.loc[X.index, SEGMENT_COL]
)

result_df = df.loc[X.index].copy()
result_df["ASSETS_POTENTIAL"] = assets_pred

display(result_df[[SEGMENT_COL, TARGET_NAME, "ASSETS_POTENTIAL"]].head())