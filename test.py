# ============================================================
# ASSETS POTENTIAL MODEL
# Active classifier + income bucket model + rich tail correction
# ============================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

TARGET_NAME = "ASSETS"
ACTIVE_THRESHOLD = 80

TEST_SIZE = 0.15
VAL_SIZE = 0.15

CORR_THRESHOLD = 0.90

# Якщо true active rate дуже малий, краще не використовувати neg/pos напряму
USE_SQRT_SCALE_POS_WEIGHT = True

# Bucket-и серед активних клієнтів
# більше bucket-ів у хвості, щоб краще ловити багатих клієнтів
BUCKET_QUANTILES = [0.00, 0.20, 0.40, 0.60, 0.75, 0.85, 0.93, 0.97, 0.99, 1.00]

# Як оцінюємо суму в bucket-і
# median — консервативно
# mean — може завищувати
# blend — компроміс
BUCKET_VALUE_MODE = "blend"

# Для top bucket даємо більше ваги mean, бо інакше багатих буде сильно занижувати
TOP_BUCKET_MEAN_WEIGHT = 0.45
NORMAL_BUCKET_MEAN_WEIGHT = 0.25

# Tail correction для найбагатших активних
USE_TAIL_REGRESSOR = True
TAIL_Q = 0.97
TAIL_MIN_ROWS = 100

# Пошук порогу активності
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.01)

# Score для вибору threshold
W_TOTAL_RATIO = 3.0
W_ACTIVE_RATE = 2.0
W_FALSE_POSITIVE = 5.0
W_MAE = 1.0
W_DISTRIBUTION = 2.0

MODEL_PATH = "assets_potential_bucket_model.joblib"


# ============================================================
# HELPERS
# ============================================================

def remove_highly_correlated_features_train_only(X_train, threshold=0.90):
    """
    Видаляємо highly correlated features тільки на train.
    Це важливо, щоб не було leakage.
    """
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    if len(num_cols) == 0:
        return []

    corr_matrix = X_train[num_cols].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper_triangle.columns
        if (upper_triangle[col] > threshold).any()
    ]

    return to_drop


def align_categories(X_train, X_val, X_test):
    """
    Фіксуємо категорії по train.
    Val/test приводимо до тих самих категорій.
    """
    cat_cols = [
        c for c in X_train.columns
        if X_train[c].dtype.name in ("object", "category")
    ]

    cat_values = {}

    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        categories = X_train[c].cat.categories

        X_val[c] = pd.Categorical(X_val[c], categories=categories)
        X_test[c] = pd.Categorical(X_test[c], categories=categories)

        cat_values[c] = list(categories)

    return X_train, X_val, X_test, cat_cols, cat_values


def make_bucket_edges(y_train_active, quantiles):
    """
    Робимо bucket edges тільки на train-active.
    """
    edges = np.quantile(y_train_active, quantiles)
    edges = np.unique(edges)

    edges[0] = -np.inf
    edges[-1] = np.inf

    if len(edges) < 3:
        raise ValueError("Too few unique bucket edges. Active target has too little variation.")

    return edges


def assign_buckets(y, edges):
    """
    Перетворюємо y в bucket labels.
    labels: 0 ... n_buckets-1
    """
    labels = pd.cut(
        y,
        bins=edges,
        labels=False,
        include_lowest=True
    )

    return labels.astype(int)


def calculate_bucket_values(y_train_active, y_train_bucket, n_buckets):
    """
    Для кожного bucket-а рахуємо representative value.
    """
    values = []

    tmp = pd.DataFrame({
        "y": np.asarray(y_train_active),
        "bucket": np.asarray(y_train_bucket)
    })

    global_median = np.median(y_train_active)

    for b in range(n_buckets):
        vals = tmp.loc[tmp["bucket"] == b, "y"].values

        if len(vals) == 0:
            values.append(global_median)
            continue

        med = np.median(vals)
        mean = np.mean(vals)

        if BUCKET_VALUE_MODE == "median":
            bucket_value = med

        elif BUCKET_VALUE_MODE == "mean":
            bucket_value = mean

        elif BUCKET_VALUE_MODE == "blend":
            if b == n_buckets - 1:
                w = TOP_BUCKET_MEAN_WEIGHT
            else:
                w = NORMAL_BUCKET_MEAN_WEIGHT

            bucket_value = (1 - w) * med + w * mean

        else:
            raise ValueError("Unknown BUCKET_VALUE_MODE")

        values.append(bucket_value)

    return np.asarray(values)


def distribution_penalty(y_true, y_pred):
    """
    Штраф за поганий match розподілу.
    Порівнюємо quantiles у log-space.
    """
    y_true_log = np.log1p(np.asarray(y_true))
    y_pred_log = np.log1p(np.asarray(y_pred))

    qs = [0.50, 0.75, 0.90, 0.95, 0.99]

    true_q = np.quantile(y_true_log, qs)
    pred_q = np.quantile(y_pred_log, qs)

    return np.mean(np.abs(true_q - pred_q))


def evaluate_final_prediction(y_true, y_pred, active_threshold=80):
    """
    Повна діагностика фінального прогнозу.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    true_total = y_true.sum()
    pred_total = y_pred.sum()

    y_true_active = (y_true >= active_threshold).astype(int)
    y_pred_active = (y_pred > 0).astype(int)

    result = {
        "mae": mean_absolute_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),

        "mae_log": mean_absolute_error(y_true_log, y_pred_log),
        "medae_log": median_absolute_error(y_true_log, y_pred_log),
        "r2_log": r2_score(y_true_log, y_pred_log),

        "true_total": true_total,
        "pred_total": pred_total,
        "total_ratio": pred_total / true_total if true_total > 0 else np.nan,

        "real_active_rate": y_true_active.mean(),
        "pred_active_rate": y_pred_active.mean(),

        "false_positive_rate": np.mean((y_true_active == 0) & (y_pred_active == 1)),
        "false_negative_rate": np.mean((y_true_active == 1) & (y_pred_active == 0)),

        "distribution_penalty": distribution_penalty(y_true, y_pred)
    }

    return result


def print_metrics(title, metrics):
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"MAE                : {metrics['mae']:,.2f}")
    print(f"MedAE              : {metrics['medae']:,.2f}")
    print(f"R2                 : {metrics['r2']:.4f}")
    print("-" * 80)
    print(f"MAE_log            : {metrics['mae_log']:.5f}")
    print(f"MedAE_log          : {metrics['medae_log']:.5f}")
    print(f"R2_log             : {metrics['r2_log']:.5f}")
    print("-" * 80)
    print(f"Real total         : {metrics['true_total']:,.2f}")
    print(f"Pred total         : {metrics['pred_total']:,.2f}")
    print(f"Pred / Real        : {metrics['total_ratio']:.4f}")
    print("-" * 80)
    print(f"Real active rate   : {metrics['real_active_rate']:.4f}")
    print(f"Pred active rate   : {metrics['pred_active_rate']:.4f}")
    print(f"False positive rate: {metrics['false_positive_rate']:.4f}")
    print(f"False negative rate: {metrics['false_negative_rate']:.4f}")
    print(f"Distribution pen.  : {metrics['distribution_penalty']:.5f}")
    print("=" * 80)


def make_bucket_report(y_true, y_pred, n_bins=10):
    tmp = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred)
    })

    tmp["true_bucket"] = pd.qcut(
        tmp["y_true"].rank(method="first"),
        q=n_bins,
        duplicates="drop"
    )

    report = tmp.groupby("true_bucket").agg(
        clients=("y_true", "size"),
        true_min=("y_true", "min"),
        true_max=("y_true", "max"),
        real_mean=("y_true", "mean"),
        pred_mean=("y_pred", "mean"),
        real_median=("y_true", "median"),
        pred_median=("y_pred", "median"),
        real_total=("y_true", "sum"),
        pred_total=("y_pred", "sum"),
    )

    report["pred_to_real_ratio"] = report["pred_total"] / report["real_total"].replace(0, np.nan)

    return report


# ============================================================
# DATA PREPARATION
# ============================================================

X_model = X.copy()
y_model = pd.Series(y).copy()

y_model = y_model.replace([np.inf, -np.inf], np.nan)
valid_target_mask = y_model.notna()

X_model = X_model.loc[valid_target_mask].copy()
y_model = y_model.loc[valid_target_mask].copy()

y_clean = np.clip(y_model, a_min=0, a_max=None)
y_binary = (y_clean >= ACTIVE_THRESHOLD).astype(int)

print("=" * 80)
print("TARGET DIAGNOSTICS")
print("=" * 80)
print(y_clean.describe())
print("-" * 80)
print(f"Active threshold : {ACTIVE_THRESHOLD}")
print(f"Active rate      : {y_binary.mean():.4f}")
print(f"Zero/low rate    : {(y_binary == 0).mean():.4f}")
print("=" * 80)


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

X_train, X_temp, y_train_raw, y_temp_raw, y_train_bin, y_temp_bin = train_test_split(
    X_model,
    y_clean,
    y_binary,
    test_size=VAL_SIZE + TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_binary
)

relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)

X_val, X_test, y_val_raw, y_test_raw, y_val_bin, y_test_bin = train_test_split(
    X_temp,
    y_temp_raw,
    y_temp_bin,
    test_size=relative_test_size,
    random_state=RANDOM_STATE,
    stratify=y_temp_bin
)

print("Split sizes:")
print(f"Train: {X_train.shape}")
print(f"Val  : {X_val.shape}")
print(f"Test : {X_test.shape}")


# ============================================================
# REMOVE HIGHLY CORRELATED FEATURES — TRAIN ONLY
# ============================================================

removed_corr_features = remove_highly_correlated_features_train_only(
    X_train,
    threshold=CORR_THRESHOLD
)

print("=" * 80)
print("CORRELATED FEATURES REMOVED")
print("=" * 80)
print(f"Removed: {len(removed_corr_features)}")
print(removed_corr_features[:50])

X_train = X_train.drop(columns=removed_corr_features, errors="ignore")
X_val = X_val.drop(columns=removed_corr_features, errors="ignore")
X_test = X_test.drop(columns=removed_corr_features, errors="ignore")


# ============================================================
# CATEGORY ALIGNMENT
# ============================================================

X_train, X_val, X_test, cat_cols, cat_values = align_categories(
    X_train.copy(),
    X_val.copy(),
    X_test.copy()
)

feature_cols = X_train.columns.tolist()

print("=" * 80)
print("FEATURES")
print("=" * 80)
print(f"Total features      : {len(feature_cols)}")
print(f"Categorical features: {len(cat_cols)}")
print(cat_cols)


# ============================================================
# STAGE 1: ACTIVE / NON-ACTIVE CLASSIFIER
# ============================================================

print("=" * 80)
print("STAGE 1: ACTIVE CLASSIFIER")
print("=" * 80)

pos = y_train_bin.sum()
neg = len(y_train_bin) - pos

if USE_SQRT_SCALE_POS_WEIGHT:
    scale_pos_weight = np.sqrt(neg / pos)
else:
    scale_pos_weight = neg / pos

print(f"Positive class: {pos}")
print(f"Negative class: {neg}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.02,

    num_leaves=31,
    max_depth=6,
    min_child_samples=50,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=1.0,
    reg_lambda=5.0,

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train,
    y_train_bin,
    eval_set=[(X_val, y_val_bin)],
    eval_metric="auc",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False)
    ]
)

prob_train = clf.predict_proba(X_train)[:, 1]
prob_val = clf.predict_proba(X_val)[:, 1]
prob_test = clf.predict_proba(X_test)[:, 1]

print(f"Classifier ROC-AUC train: {roc_auc_score(y_train_bin, prob_train):.4f}")
print(f"Classifier ROC-AUC val  : {roc_auc_score(y_val_bin, prob_val):.4f}")
print(f"Classifier ROC-AUC test : {roc_auc_score(y_test_bin, prob_test):.4f}")


# ============================================================
# STAGE 2: BUCKET MODEL AMONG ACTIVE CLIENTS
# ============================================================

print("=" * 80)
print("STAGE 2: ACTIVE CLIENT BUCKET MODEL")
print("=" * 80)

mask_train_active = (y_train_raw >= ACTIVE_THRESHOLD).values
mask_val_active = (y_val_raw >= ACTIVE_THRESHOLD).values
mask_test_active = (y_test_raw >= ACTIVE_THRESHOLD).values

X_train_active = X_train.loc[mask_train_active].copy()
X_val_active = X_val.loc[mask_val_active].copy()
X_test_active = X_test.loc[mask_test_active].copy()

y_train_active = y_train_raw.loc[mask_train_active].copy()
y_val_active = y_val_raw.loc[mask_val_active].copy()
y_test_active = y_test_raw.loc[mask_test_active].copy()

bucket_edges = make_bucket_edges(y_train_active, BUCKET_QUANTILES)
n_buckets = len(bucket_edges) - 1

y_train_bucket = assign_buckets(y_train_active, bucket_edges)
y_val_bucket = assign_buckets(y_val_active, bucket_edges)
y_test_bucket = assign_buckets(y_test_active, bucket_edges)

bucket_values = calculate_bucket_values(
    y_train_active=y_train_active,
    y_train_bucket=y_train_bucket,
    n_buckets=n_buckets
)

bucket_table = pd.DataFrame({
    "bucket": np.arange(n_buckets),
    "left_edge": bucket_edges[:-1],
    "right_edge": bucket_edges[1:],
    "bucket_value": bucket_values
})

print("Bucket table:")
display(bucket_table)

bucket_clf = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=n_buckets,

    n_estimators=3000,
    learning_rate=0.02,

    num_leaves=31,
    max_depth=6,
    min_child_samples=40,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=1.0,
    reg_lambda=5.0,

    class_weight="balanced",

    random_state=RANDOM_STATE,
    n_jobs=-1
)

bucket_clf.fit(
    X_train_active,
    y_train_bucket,
    eval_set=[(X_val_active, y_val_bucket)],
    eval_metric="multi_logloss",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False)
    ]
)

bucket_pred_val_active = bucket_clf.predict(X_val_active)
bucket_pred_test_active = bucket_clf.predict(X_test_active)

print("Bucket classification report, VAL active clients:")
print(classification_report(y_val_bucket, bucket_pred_val_active))

print("Bucket classification report, TEST active clients:")
print(classification_report(y_test_bucket, bucket_pred_test_active))


# ============================================================
# STAGE 3: OPTIONAL TAIL REGRESSOR FOR RICH CLIENTS
# ============================================================

tail_reg = None
tail_threshold = None
tail_cap = None

if USE_TAIL_REGRESSOR:
    tail_threshold = np.quantile(y_train_active, TAIL_Q)
    tail_mask_train = y_train_active >= tail_threshold
    tail_mask_val = y_val_active >= tail_threshold

    print("=" * 80)
    print("STAGE 3: TAIL REGRESSOR")
    print("=" * 80)
    print(f"Tail threshold q={TAIL_Q}: {tail_threshold:,.2f}")
    print(f"Tail train rows: {tail_mask_train.sum()}")

    if tail_mask_train.sum() >= TAIL_MIN_ROWS:
        X_train_tail = X_train_active.loc[tail_mask_train].copy()
        y_train_tail = y_train_active.loc[tail_mask_train].copy()

        X_val_tail = X_val_active.loc[tail_mask_val].copy()
        y_val_tail = y_val_active.loc[tail_mask_val].copy()

        tail_cap = np.quantile(y_train_tail, 0.995)

        y_train_tail_log = np.log1p(np.clip(y_train_tail, 0, tail_cap))

        if len(y_val_tail) > 0:
            y_val_tail_log = np.log1p(np.clip(y_val_tail, 0, tail_cap))
            eval_set_tail = [(X_val_tail, y_val_tail_log)]
        else:
            eval_set_tail = None

        tail_reg = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",

            n_estimators=2000,
            learning_rate=0.02,

            num_leaves=15,
            max_depth=4,
            min_child_samples=30,

            subsample=0.8,
            colsample_bytree=0.8,

            reg_alpha=2.0,
            reg_lambda=10.0,

            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        if eval_set_tail is not None and len(y_val_tail) >= 20:
            tail_reg.fit(
                X_train_tail,
                y_train_tail_log,
                eval_set=eval_set_tail,
                eval_metric="mae",
                categorical_feature=cat_cols,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=150, verbose=False)
                ]
            )
        else:
            tail_reg.fit(
                X_train_tail,
                y_train_tail_log,
                categorical_feature=cat_cols
            )

        tail_pred_train = np.expm1(tail_reg.predict(X_train_tail))
        tail_pred_train = np.clip(tail_pred_train, 0, tail_cap)

        print(f"Tail MAE train: {mean_absolute_error(y_train_tail, tail_pred_train):,.2f}")
        print(f"Tail MedAE train: {median_absolute_error(y_train_tail, tail_pred_train):,.2f}")

        if len(y_val_tail) > 0:
            tail_pred_val = np.expm1(tail_reg.predict(X_val_tail))
            tail_pred_val = np.clip(tail_pred_val, 0, tail_cap)
            print(f"Tail MAE val: {mean_absolute_error(y_val_tail, tail_pred_val):,.2f}")
            print(f"Tail MedAE val: {median_absolute_error(y_val_tail, tail_pred_val):,.2f}")

    else:
        print("Not enough tail rows. Tail regressor skipped.")


# ============================================================
# PREDICTION FUNCTION FOR VALIDATION / TEST
# ============================================================

def predict_assets_potential_raw(
    X_data,
    active_prob,
    active_threshold,
    use_tail=True
):
    """
    Повертає прогноз до threshold search.
    active_threshold тут — поріг імовірності active / non-active.
    """
    bucket_proba = bucket_clf.predict_proba(X_data)

    bucket_argmax = np.argmax(bucket_proba, axis=1)
    bucket_pred = bucket_values[bucket_argmax]

    # Expected value можна дивитися як альтернативу,
    # але для match розподілу краще bucket_argmax.
    bucket_ev = bucket_proba @ bucket_values

    # Основний прогноз — bucket_argmax.
    amount_pred = bucket_pred.copy()

    # Tail correction тільки для тих, кого bucket model віднесла у top bucket.
    if use_tail and tail_reg is not None:
        top_bucket = n_buckets - 1
        top_candidates = bucket_argmax == top_bucket

        if top_candidates.sum() > 0:
            tail_pred = np.expm1(tail_reg.predict(X_data.loc[top_candidates]))
            tail_pred = np.clip(tail_pred, 0, tail_cap)

            # blend, щоб tail regressor не розганяв занадто сильно
            amount_pred[top_candidates] = (
                0.50 * amount_pred[top_candidates] +
                0.50 * tail_pred
            )

    final_pred = np.where(active_prob >= active_threshold, amount_pred, 0)
    final_pred = np.clip(final_pred, 0, None)

    return final_pred, bucket_argmax, bucket_ev


# ============================================================
# THRESHOLD SEARCH ON VALIDATION
# ============================================================

print("=" * 80)
print("THRESHOLD SEARCH")
print("=" * 80)

threshold_rows = []

for th in THRESHOLD_GRID:
    pred_val_th, _, _ = predict_assets_potential_raw(
        X_data=X_val,
        active_prob=prob_val,
        active_threshold=th,
        use_tail=True
    )

    metrics = evaluate_final_prediction(
        y_true=y_val_raw,
        y_pred=pred_val_th,
        active_threshold=ACTIVE_THRESHOLD
    )

    score = (
        W_TOTAL_RATIO * abs(metrics["total_ratio"] - 1.0)
        + W_ACTIVE_RATE * abs(metrics["pred_active_rate"] - metrics["real_active_rate"])
        + W_FALSE_POSITIVE * metrics["false_positive_rate"]
        + W_MAE * (metrics["mae"] / max(1, np.median(threshold_rows[-10:][0]["mae"]) if len(threshold_rows) > 10 else metrics["mae"]))
        + W_DISTRIBUTION * metrics["distribution_penalty"]
    )

    row = {
        "threshold": th,
        "score": score,
        **metrics
    }

    threshold_rows.append(row)

threshold_results = pd.DataFrame(threshold_rows)

best_row = threshold_results.sort_values("score").iloc[0]
BEST_ACTIVE_THRESHOLD = float(best_row["threshold"])

print("Best threshold row:")
display(best_row.to_frame().T)

print("Top 10 thresholds:")
display(threshold_results.sort_values("score").head(10))


# ============================================================
# FINAL PREDICTIONS
# ============================================================

pred_train, train_bucket_idx, train_bucket_ev = predict_assets_potential_raw(
    X_data=X_train,
    active_prob=prob_train,
    active_threshold=BEST_ACTIVE_THRESHOLD,
    use_tail=True
)

pred_val, val_bucket_idx, val_bucket_ev = predict_assets_potential_raw(
    X_data=X_val,
    active_prob=prob_val,
    active_threshold=BEST_ACTIVE_THRESHOLD,
    use_tail=True
)

pred_test, test_bucket_idx, test_bucket_ev = predict_assets_potential_raw(
    X_data=X_test,
    active_prob=prob_test,
    active_threshold=BEST_ACTIVE_THRESHOLD,
    use_tail=True
)


# ============================================================
# FINAL METRICS
# ============================================================

train_metrics = evaluate_final_prediction(
    y_true=y_train_raw,
    y_pred=pred_train,
    active_threshold=ACTIVE_THRESHOLD
)

val_metrics = evaluate_final_prediction(
    y_true=y_val_raw,
    y_pred=pred_val,
    active_threshold=ACTIVE_THRESHOLD
)

test_metrics = evaluate_final_prediction(
    y_true=y_test_raw,
    y_pred=pred_test,
    active_threshold=ACTIVE_THRESHOLD
)

print_metrics("TRAIN FINAL METRICS", train_metrics)
print_metrics("VALIDATION FINAL METRICS", val_metrics)
print_metrics("TEST FINAL METRICS", test_metrics)


# ============================================================
# STAGE 1 CLASSIFICATION DIAGNOSTICS AT BEST THRESHOLD
# ============================================================

val_active_pred = (prob_val >= BEST_ACTIVE_THRESHOLD).astype(int)
test_active_pred = (prob_test >= BEST_ACTIVE_THRESHOLD).astype(int)

print("=" * 80)
print("ACTIVE CLASSIFIER REPORT — VAL")
print("=" * 80)
print(classification_report(y_val_bin, val_active_pred))
print("Confusion matrix:")
print(confusion_matrix(y_val_bin, val_active_pred))

print("=" * 80)
print("ACTIVE CLASSIFIER REPORT — TEST")
print("=" * 80)
print(classification_report(y_test_bin, test_active_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test_bin, test_active_pred))


# ============================================================
# DISTRIBUTION / BUCKET DIAGNOSTICS
# ============================================================

print("=" * 80)
print("VALIDATION BUCKET REPORT BY TRUE TARGET DECILES")
print("=" * 80)
val_bucket_report = make_bucket_report(y_val_raw, pred_val, n_bins=10)
display(val_bucket_report)

print("=" * 80)
print("TEST BUCKET REPORT BY TRUE TARGET DECILES")
print("=" * 80)
test_bucket_report = make_bucket_report(y_test_raw, pred_test, n_bins=10)
display(test_bucket_report)


# ============================================================
# SEGMENT DIAGNOSTICS
# ============================================================

def make_segment_report(X_data, y_true, y_pred, active_prob):
    if "FIRM_TYPE" in X_data.columns:
        segment = X_data["FIRM_TYPE"].astype(str).values
    else:
        segment = np.array(["UNKNOWN"] * len(X_data))

    tmp = pd.DataFrame({
        "FIRM_TYPE": segment,
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "active_prob": np.asarray(active_prob)
    })

    report = tmp.groupby("FIRM_TYPE").agg(
        clients=("y_true", "size"),
        real_total=("y_true", "sum"),
        pred_total=("y_pred", "sum"),
        real_mean=("y_true", "mean"),
        pred_mean=("y_pred", "mean"),
        real_median=("y_true", "median"),
        pred_median=("y_pred", "median"),
        real_active_rate=("y_true", lambda x: (x >= ACTIVE_THRESHOLD).mean()),
        pred_active_rate=("y_pred", lambda x: (x > 0).mean()),
        avg_active_prob=("active_prob", "mean")
    )

    report["pred_to_real_ratio"] = report["pred_total"] / report["real_total"].replace(0, np.nan)

    return report


print("=" * 80)
print("SEGMENT REPORT — VAL")
print("=" * 80)
val_segment_report = make_segment_report(X_val, y_val_raw, pred_val, prob_val)
display(val_segment_report)

print("=" * 80)
print("SEGMENT REPORT — TEST")
print("=" * 80)
test_segment_report = make_segment_report(X_test, y_test_raw, pred_test, prob_test)
display(test_segment_report)


# ============================================================
# PLOTS
# ============================================================

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

y_val_log = np.log1p(y_val_raw)
pred_val_log = np.log1p(pred_val)

axes[0].scatter(y_val_log, pred_val_log, alpha=0.25, s=15)
mn = 0
mx = max(float(np.max(y_val_log)), float(np.max(pred_val_log))) + 1
axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=2)
axes[0].set_xlabel("True ASSETS log1p")
axes[0].set_ylabel("Predicted ASSETS_POTENTIAL log1p")
axes[0].set_title("Validation: True vs Predicted")

sns.kdeplot(y_val_log, label="True Distribution", fill=True, alpha=0.15, ax=axes[1])
sns.kdeplot(pred_val_log, label="Predicted Distribution", linestyle="--", ax=axes[1])
axes[1].set_title("Validation: Distribution Match")
axes[1].set_xlabel("log1p(ASSETS)")
axes[1].legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(threshold_results["threshold"], threshold_results["total_ratio"], label="Pred / Real total")
plt.axhline(1.0, linestyle="--")
plt.axvline(BEST_ACTIVE_THRESHOLD, linestyle="--", label=f"Best threshold = {BEST_ACTIVE_THRESHOLD:.2f}")
plt.title("Threshold Search: Total Ratio")
plt.xlabel("Active probability threshold")
plt.ylabel("Pred total / Real total")
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(threshold_results["threshold"], threshold_results["pred_active_rate"], label="Pred active rate")
plt.axhline(threshold_results["real_active_rate"].iloc[0], linestyle="--", label="Real active rate")
plt.axvline(BEST_ACTIVE_THRESHOLD, linestyle="--", label=f"Best threshold = {BEST_ACTIVE_THRESHOLD:.2f}")
plt.title("Threshold Search: Active Rate")
plt.xlabel("Active probability threshold")
plt.ylabel("Rate")
plt.legend()
plt.show()


# ============================================================
# SAVE MODEL PACKAGE
# ============================================================

model_package = {
    "model_type": "assets_potential_bucket_model",
    "active_threshold_target": ACTIVE_THRESHOLD,
    "best_active_probability_threshold": BEST_ACTIVE_THRESHOLD,

    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "cat_values": cat_values,
    "removed_corr_features": removed_corr_features,

    "active_classifier": clf,

    "bucket_classifier": bucket_clf,
    "bucket_edges": bucket_edges,
    "bucket_values": bucket_values,
    "bucket_table": bucket_table,

    "use_tail_regressor": USE_TAIL_REGRESSOR,
    "tail_regressor": tail_reg,
    "tail_threshold": tail_threshold,
    "tail_cap": tail_cap,

    "config": {
        "RANDOM_STATE": RANDOM_STATE,
        "ACTIVE_THRESHOLD": ACTIVE_THRESHOLD,
        "BUCKET_QUANTILES": BUCKET_QUANTILES,
        "BUCKET_VALUE_MODE": BUCKET_VALUE_MODE,
        "TOP_BUCKET_MEAN_WEIGHT": TOP_BUCKET_MEAN_WEIGHT,
        "NORMAL_BUCKET_MEAN_WEIGHT": NORMAL_BUCKET_MEAN_WEIGHT,
        "USE_TAIL_REGRESSOR": USE_TAIL_REGRESSOR,
        "TAIL_Q": TAIL_Q,
        "BEST_ACTIVE_THRESHOLD": BEST_ACTIVE_THRESHOLD
    }
}

joblib.dump(model_package, MODEL_PATH)

print("=" * 80)
print(f"Model saved to: {MODEL_PATH}")
print("=" * 80)