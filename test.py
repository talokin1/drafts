import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    r2_score
)


RANDOM_STATE = 42

ID_COL = "IDENTIFYCODE"
SEGMENT_COL = "FIRM_TYPE"


# ВАЖЛИВО:
X = X_.copy()
y = df["CURR_ACC"]

BUCKET_EDGES = [
    -0.001,
    100,
    500,
    1_500,
    5_000,
    15_000,
    50_000,
    150_000,
    np.inf
]

BUCKET_LABELS = [
    "0-100",
    "100-500",
    "500-1.5k",
    "1.5k-5k",
    "5k-15k",
    "15k-50k",
    "50k-150k",
    "150k+"
]

N_BUCKETS = len(BUCKET_LABELS)














def make_target_buckets(y, bucket_edges, bucket_labels):
    y_clean = pd.Series(y).clip(lower=0)

    buckets = pd.cut(
        y_clean,
        bins=bucket_edges,
        labels=False,
        include_lowest=True
    )

    buckets = buckets.fillna(0).astype(int)

    bucket_names = pd.Series(
        [bucket_labels[i] for i in buckets],
        index=y_clean.index
    )

    return buckets, bucket_names


def prepare_categorical_train_valid(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    cat_cols = [
        c for c in X_train.columns
        if X_train[c].dtype.name in ("object", "category")
    ]

    cat_values = {}

    for c in cat_cols:
        categories = (
            pd.Series(X_train[c].astype("object"))
            .dropna()
            .unique()
            .tolist()
        )

        cat_values[c] = categories

        X_train[c] = pd.Categorical(X_train[c], categories=categories)
        X_val[c] = pd.Categorical(X_val[c], categories=categories)

    return X_train, X_val, cat_cols, cat_values


def prepare_categorical_inference(X_new, feature_cols, cat_cols, cat_values):
    X_new = X_new.copy()

    missing_cols = [c for c in feature_cols if c not in X_new.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in inference dataframe: {missing_cols}")

    X_new = X_new[feature_cols].copy()

    for c in cat_cols:
        X_new[c] = pd.Categorical(X_new[c], categories=cat_values[c])

    return X_new


def build_bucket_values(y_train, y_train_bucket, bucket_labels):
    tmp = pd.DataFrame({
        "target": pd.Series(y_train).clip(lower=0),
        "bucket": y_train_bucket
    })

    bucket_values = {}

    global_positive_median = tmp.loc[tmp["target"] > 0, "target"].median()

    if pd.isna(global_positive_median):
        global_positive_median = 0

    for bucket_id, bucket_name in enumerate(bucket_labels):
        vals = tmp.loc[tmp["bucket"] == bucket_id, "target"]

        if len(vals) == 0:
            bucket_values[bucket_id] = global_positive_median
        else:
            # Для 0-100 краще медіана, для інших теж медіана стабільніша за mean
            bucket_values[bucket_id] = vals.median()

    return bucket_values


def expected_value_from_proba(proba, bucket_values):
    bucket_value_array = np.array([
        bucket_values[i] for i in range(len(bucket_values))
    ])

    pred = proba @ bucket_value_array
    pred = np.clip(pred, 0, None)

    return pred


def safe_ratio(num, den, default=1.0):
    if den == 0 or pd.isna(den):
        return default
    return num / den











df_model = X.copy()
y_clean = pd.Series(y, index=X.index).clip(lower=0)

y_bucket, y_bucket_name = make_target_buckets(
    y_clean,
    bucket_edges=BUCKET_EDGES,
    bucket_labels=BUCKET_LABELS
)

print("Target describe:")
print(y_clean.describe())

print("\nBucket distribution:")
print(y_bucket_name.value_counts().sort_index())
print(y_bucket.value_counts(normalize=True).sort_index())

X_train, X_val, y_train_raw, y_val_raw, y_train_bucket, y_val_bucket = train_test_split(
    df_model,
    y_clean,
    y_bucket,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_bucket
)

X_train, X_val, cat_cols, cat_values = prepare_categorical_train_valid(
    X_train,
    X_val
)

feature_cols = X_train.columns.tolist()

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("cat_cols:", cat_cols)







bucket_values = build_bucket_values(
    y_train=y_train_raw,
    y_train_bucket=y_train_bucket,
    bucket_labels=BUCKET_LABELS
)

bucket_value_table = pd.DataFrame({
    "bucket_id": list(bucket_values.keys()),
    "bucket_name": [BUCKET_LABELS[i] for i in bucket_values.keys()],
    "bucket_value": list(bucket_values.values())
})

bucket_value_table


def build_bucket_class_weights(y_bucket):
    counts = pd.Series(y_bucket).value_counts().sort_index()
    total = counts.sum()
    n_classes = len(counts)

    class_weights = {}

    for cls, count in counts.items():
        class_weights[int(cls)] = total / (n_classes * count)

    return class_weights


def build_segment_weights(X_part):
    weights = pd.Series(1.0, index=X_part.index, dtype=float)

    if SEGMENT_COL in X_part.columns:
        seg = X_part[SEGMENT_COL].astype(str)

        weights.loc[seg.eq("MICRO")] = 1.0
        weights.loc[seg.eq("SMALL")] = 1.2
        weights.loc[seg.eq("MEDIUM")] = 1.5
        weights.loc[seg.eq("LARGE")] = 2.5

    return weights


class_weights = build_bucket_class_weights(y_train_bucket)

bucket_weight = pd.Series(y_train_bucket).map(class_weights).values
segment_weight = build_segment_weights(X_train).values

sample_weight = bucket_weight * segment_weight

print("Class weights:")
print(class_weights)










bucket_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=N_BUCKETS,
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


print("Training Bucket Expected Value Model...")

bucket_model.fit(
    X_train,
    y_train_bucket,
    sample_weight=sample_weight,
    eval_set=[(X_val, y_val_bucket)],
    eval_metric="multi_logloss",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

train_proba = bucket_model.predict_proba(X_train)
val_proba = bucket_model.predict_proba(X_val)

train_bucket_pred = np.argmax(train_proba, axis=1)
val_bucket_pred = np.argmax(val_proba, axis=1)

train_expected_raw = expected_value_from_proba(train_proba, bucket_values)
val_expected_raw = expected_value_from_proba(val_proba, bucket_values)




print("=" * 80)
print("[Bucket Classification Metrics]")
print("-" * 80)

print(f"Train accuracy: {accuracy_score(y_train_bucket, train_bucket_pred):.4f}")
print(f"Val accuracy  : {accuracy_score(y_val_bucket, val_bucket_pred):.4f}")

print(f"Train F1 macro: {f1_score(y_train_bucket, train_bucket_pred, average='macro'):.4f}")
print(f"Val F1 macro  : {f1_score(y_val_bucket, val_bucket_pred, average='macro'):.4f}")

print("-" * 80)
print(classification_report(
    y_val_bucket,
    val_bucket_pred,
    target_names=BUCKET_LABELS
))

print("=" * 80)







def build_segment_calibration(
    X_val,
    y_true,
    y_pred,
    segment_col=SEGMENT_COL,
    factor_min=0.25,
    factor_max=3.0
):
    tmp = pd.DataFrame(index=X_val.index)

    if segment_col in X_val.columns:
        tmp[segment_col] = X_val[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["true"] = np.array(y_true)
    tmp["pred"] = np.array(y_pred)

    global_factor = safe_ratio(
        tmp["true"].sum(),
        tmp["pred"].sum(),
        default=1.0
    )

    calibration = (
        tmp.groupby(segment_col)
        .agg(
            n=("true", "size"),
            true_sum=("true", "sum"),
            pred_sum=("pred", "sum"),
            true_mean=("true", "mean"),
            pred_mean=("pred", "mean"),
            true_median=("true", "median"),
            pred_median=("pred", "median")
        )
        .reset_index()
    )

    calibration["factor"] = (
        calibration["true_sum"] /
        calibration["pred_sum"].replace(0, np.nan)
    )

    calibration["factor"] = (
        calibration["factor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_factor)
        .clip(factor_min, factor_max)
    )

    return calibration, global_factor


def apply_segment_calibration(
    X_part,
    y_pred,
    calibration,
    global_factor,
    segment_col=SEGMENT_COL
):
    tmp = pd.DataFrame(index=X_part.index)

    if segment_col in X_part.columns:
        tmp[segment_col] = X_part[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp = tmp.merge(
        calibration[[segment_col, "factor"]],
        on=segment_col,
        how="left"
    )

    tmp["factor"] = tmp["factor"].fillna(global_factor)
    tmp["factor"] = tmp["factor"].fillna(1.0)

    pred_calibrated = np.array(y_pred) * tmp["factor"].values
    pred_calibrated = np.clip(pred_calibrated, 0, None)

    return pred_calibrated, tmp["factor"].values

segment_calibration, global_calibration_factor = build_segment_calibration(
    X_val=X_val,
    y_true=y_val_raw,
    y_pred=val_expected_raw,
    segment_col=SEGMENT_COL,
    factor_min=0.25,
    factor_max=3.0
)

display(segment_calibration)
print("Global calibration factor:", global_calibration_factor)



train_expected_calibrated, train_calibration_factor = apply_segment_calibration(
    X_part=X_train,
    y_pred=train_expected_raw,
    calibration=segment_calibration,
    global_factor=global_calibration_factor,
    segment_col=SEGMENT_COL
)

val_expected_calibrated, val_calibration_factor = apply_segment_calibration(
    X_part=X_val,
    y_pred=val_expected_raw,
    calibration=segment_calibration,
    global_factor=global_calibration_factor,
    segment_col=SEGMENT_COL
)




def build_caps_by_segment(
    X_train,
    y_train,
    segment_col=SEGMENT_COL,
    cap_quantile=0.995
):
    tmp = pd.DataFrame(index=X_train.index)

    if segment_col in X_train.columns:
        tmp[segment_col] = X_train[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["target"] = np.array(y_train)

    caps_by_segment = (
        tmp.groupby(segment_col)["target"]
        .quantile(cap_quantile)
        .to_dict()
    )

    global_cap = tmp["target"].quantile(cap_quantile)

    return caps_by_segment, global_cap


def apply_caps(
    X_part,
    pred,
    caps_by_segment,
    global_cap,
    segment_col=SEGMENT_COL
):
    pred = np.array(pred, dtype=float)

    caps = np.full(len(pred), global_cap, dtype=float)

    if segment_col in X_part.columns:
        segments = X_part[segment_col].astype(str).values

        for i, seg in enumerate(segments):
            caps[i] = caps_by_segment.get(seg, global_cap)

    pred_capped = np.minimum(pred, caps)
    pred_capped = np.clip(pred_capped, 0, None)

    return pred_capped, caps


caps_by_segment, global_cap = build_caps_by_segment(
    X_train=X_train,
    y_train=y_train_raw,
    segment_col=SEGMENT_COL,
    cap_quantile=0.995
)

print("global_cap:", global_cap)
print("caps_by_segment:", caps_by_segment)


train_final_pred, train_cap_used = apply_caps(
    X_part=X_train,
    pred=train_expected_calibrated,
    caps_by_segment=caps_by_segment,
    global_cap=global_cap,
    segment_col=SEGMENT_COL
)

val_final_pred, val_cap_used = apply_caps(
    X_part=X_val,
    pred=val_expected_calibrated,
    caps_by_segment=caps_by_segment,
    global_cap=global_cap,
    segment_col=SEGMENT_COL
)










def regression_metrics(y_true, y_pred, title):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.clip(y_pred, 0, None))

    eps = 1e-9

    print("=" * 80)
    print(title)
    print("-" * 80)
    print(f"MAE       : {mean_absolute_error(y_true, y_pred):,.4f}")
    print(f"MedAE     : {median_absolute_error(y_true, y_pred):,.4f}")
    print(f"R2        : {r2_score(y_true, y_pred):,.4f}")
    print(f"MAE_log   : {mean_absolute_error(y_true_log, y_pred_log):,.4f}")
    print(f"MedAE_log : {median_absolute_error(y_true_log, y_pred_log):,.4f}")
    print(f"R2_log    : {r2_score(y_true_log, y_pred_log):,.4f}")
    print("-" * 80)
    print(f"sum_true  : {y_true.sum():,.4f}")
    print(f"sum_pred  : {y_pred.sum():,.4f}")
    print(f"sum_ratio : {y_pred.sum() / max(y_true.sum(), eps):,.4f}")
    print(f"mean_true : {y_true.mean():,.4f}")
    print(f"mean_pred : {y_pred.mean():,.4f}")
    print(f"bias_mean : {np.mean(y_pred - y_true):,.4f}")
    print(f"bias_sum  : {(y_pred.sum() - y_true.sum()):,.4f}")
    print("=" * 80)


regression_metrics(
    y_train_raw,
    train_final_pred,
    "[Train] Bucket Expected Value Model"
)

regression_metrics(
    y_val_raw,
    val_final_pred,
    "[Validation] Bucket Expected Value Model"
)

validation_results = pd.DataFrame({
    ID_COL: X_val.index,
    "True_Value": y_val_raw.values,
    "True_Bucket_ID": y_val_bucket.values,
    "True_Bucket": [BUCKET_LABELS[i] for i in y_val_bucket.values],
    "Pred_Bucket_ID": val_bucket_pred,
    "Pred_Bucket": [BUCKET_LABELS[i] for i in val_bucket_pred],
    "Expected_Raw": val_expected_raw,
    "Calibration_Factor": val_calibration_factor,
    "Expected_Calibrated": val_expected_calibrated,
    "Cap_Used": val_cap_used,
    "Predicted": val_final_pred
}, index=X_val.index)

if SEGMENT_COL in X_val.columns:
    validation_results[SEGMENT_COL] = X_val[SEGMENT_COL].astype(str).values
else:
    validation_results[SEGMENT_COL] = "ALL"

# probabilities per bucket
for i, label in enumerate(BUCKET_LABELS):
    validation_results[f"P_BUCKET_{i}_{label}"] = val_proba[:, i]

validation_results["Error"] = validation_results["Predicted"] - validation_results["True_Value"]
validation_results["Abs_Error"] = validation_results["Error"].abs()
validation_results["Ratio"] = (
    validation_results["Predicted"] /
    validation_results["True_Value"].replace(0, np.nan)
)

round_cols = [
    "Expected_Raw",
    "Calibration_Factor",
    "Expected_Calibrated",
    "Cap_Used",
    "Predicted",
    "Error",
    "Abs_Error",
    "Ratio"
]

for c in round_cols:
    validation_results[c] = validation_results[c].round(4)

prob_cols = [c for c in validation_results.columns if c.startswith("P_BUCKET_")]

for c in prob_cols:
    validation_results[c] = validation_results[c].round(4)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))

sns.kdeplot(
    np.log1p(validation_results["True_Value"]),
    label="True",
    fill=True,
    alpha=0.25
)

sns.kdeplot(
    np.log1p(validation_results["Predicted"]),
    label="Predicted",
    linestyle="--"
)

plt.xlabel("log1p income")
plt.ylabel("Density")
plt.title("Bucket Expected Value Model: True vs Predicted Distribution")
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))

plt.scatter(
    np.log1p(validation_results["True_Value"]),
    np.log1p(validation_results["Predicted"]),
    alpha=0.25,
    s=15
)

mx = max(
    np.log1p(validation_results["True_Value"]).max(),
    np.log1p(validation_results["Predicted"]).max()
)

plt.plot([0, mx], [0, mx], "--")

plt.xlabel("True log1p income")
plt.ylabel("Predicted log1p income")
plt.title("Bucket Expected Value Model: True vs Predicted")
plt.show()


plt.figure(figsize=(10, 6))


cm = confusion_matrix(y_val_bucket, val_bucket_pred)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label}" for label in BUCKET_LABELS],
    columns=[f"pred_{label}" for label in BUCKET_LABELS]
)

cm_df

sns.heatmap(
    cm_df,
    annot=False,
    cmap="Blues"
)

plt.title("Bucket Confusion Matrix")
plt.xlabel("Predicted bucket")
plt.ylabel("True bucket")
plt.show()



model_artifacts = {
    "model_type": "bucket_expected_value",

    "bucket_model": bucket_model,

    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "cat_values": cat_values,

    "bucket_edges": BUCKET_EDGES,
    "bucket_labels": BUCKET_LABELS,
    "n_buckets": N_BUCKETS,
    "bucket_values": bucket_values,

    "segment_col": SEGMENT_COL,

    "segment_calibration": segment_calibration,
    "global_calibration_factor": global_calibration_factor,

    "caps_by_segment": caps_by_segment,
    "global_cap": global_cap
}

joblib.dump(model_artifacts, MODEL_PATH)

print(f"Saved to: {MODEL_PATH}")























import numpy as np
import pandas as pd
import joblib


def prepare_categorical_inference(X_new, feature_cols, cat_cols, cat_values):
    X_new = X_new.copy()

    missing_cols = [c for c in feature_cols if c not in X_new.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in inference dataframe: {missing_cols}")

    X_new = X_new[feature_cols].copy()

    for c in cat_cols:
        X_new[c] = pd.Categorical(X_new[c], categories=cat_values[c])

    return X_new


def expected_value_from_proba(proba, bucket_values):
    bucket_value_array = np.array([
        bucket_values[i] for i in range(len(bucket_values))
    ])

    pred = proba @ bucket_value_array
    pred = np.clip(pred, 0, None)

    return pred


def apply_segment_calibration_inference(
    df_raw,
    pred,
    segment_calibration,
    global_calibration_factor,
    segment_col
):
    tmp = pd.DataFrame(index=df_raw.index)

    if segment_col in df_raw.columns:
        tmp[segment_col] = df_raw[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp = tmp.merge(
        segment_calibration[[segment_col, "factor"]],
        on=segment_col,
        how="left"
    )

    tmp["factor"] = tmp["factor"].fillna(global_calibration_factor)
    tmp["factor"] = tmp["factor"].fillna(1.0)

    pred_calibrated = pred * tmp["factor"].values
    pred_calibrated = np.clip(pred_calibrated, 0, None)

    return pred_calibrated, tmp["factor"].values


def apply_caps_inference(
    df_raw,
    pred,
    caps_by_segment,
    global_cap,
    segment_col
):
    pred = np.array(pred, dtype=float)

    caps = np.full(len(pred), global_cap, dtype=float)

    if segment_col in df_raw.columns:
        segments = df_raw[segment_col].astype(str).values

        for i, seg in enumerate(segments):
            caps[i] = caps_by_segment.get(seg, global_cap)

    pred_capped = np.minimum(pred, caps)
    pred_capped = np.clip(pred_capped, 0, None)

    return pred_capped, caps


def predict_bucket_expected_income(df_new, model_path):
    artifacts = joblib.load(model_path)

    bucket_model = artifacts["bucket_model"]

    feature_cols = artifacts["feature_cols"]
    cat_cols = artifacts["cat_cols"]
    cat_values = artifacts["cat_values"]

    bucket_labels = artifacts["bucket_labels"]
    bucket_values = artifacts["bucket_values"]

    segment_col = artifacts["segment_col"]

    segment_calibration = artifacts["segment_calibration"]
    global_calibration_factor = artifacts["global_calibration_factor"]

    caps_by_segment = artifacts["caps_by_segment"]
    global_cap = artifacts["global_cap"]

    df_raw = df_new.copy()

    X_inf = prepare_categorical_inference(
        X_new=df_raw,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        cat_values=cat_values
    )

    proba = bucket_model.predict_proba(X_inf)

    pred_bucket_id = np.argmax(proba, axis=1)
    pred_bucket_name = [bucket_labels[i] for i in pred_bucket_id]

    expected_raw = expected_value_from_proba(proba, bucket_values)

    expected_calibrated, calibration_factor = apply_segment_calibration_inference(
        df_raw=df_raw,
        pred=expected_raw,
        segment_calibration=segment_calibration,
        global_calibration_factor=global_calibration_factor,
        segment_col=segment_col
    )

    final_pred, cap_used = apply_caps_inference(
        df_raw=df_raw,
        pred=expected_calibrated,
        caps_by_segment=caps_by_segment,
        global_cap=global_cap,
        segment_col=segment_col
    )

    result = df_raw.copy()

    result["PRED_BUCKET_ID"] = pred_bucket_id
    result["PRED_BUCKET"] = pred_bucket_name

    result["LIABILITIES_EXPECTED_RAW"] = expected_raw
    result["CALIBRATION_FACTOR"] = calibration_factor
    result["LIABILITIES_EXPECTED_CALIBRATED"] = expected_calibrated
    result["CAP_USED"] = cap_used
    result["LIABILITIES_POTENTIAL"] = final_pred

    for i, label in enumerate(bucket_labels):
        result[f"P_BUCKET_{i}_{label}"] = proba[:, i]

    return result


df_pred = predict_bucket_expected_income(
    df_new=df,
    model_path=MODEL_PATH
)

df_pred.head()

df["LIABILITIES_POTENTIAL"] = df_pred["LIABILITIES_POTENTIAL"].values