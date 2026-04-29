import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    r2_score
)

sns.set_theme(style="whitegrid")


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

TARGET_NAME = "CURR_ACC"   # зміни, якщо інший таргет

# Межі bucket-ів
# 0: y < 100
# 1: 100 <= y < 500
# 2: 500 <= y < 2000
# 3: 2000 <= y < 10000
# 4: 10000 <= y < 50000
# 5: y >= 50000

BUCKET_BINS = [-np.inf, 100, 500, 2000, 10000, 50000, np.inf]

BUCKET_LABELS = [
    "zero_or_low",
    "very_low",
    "low",
    "medium",
    "high",
    "very_high"
]

ZERO_BUCKET_ID = 0

TEST_SIZE = 0.20
VAL_SIZE = 0.20

# Якщо max probability для zero bucket найбільша — ставимо 0
ZERO_IF_ZERO_BUCKET_IS_TOP = True

# Додатковий жорсткий фільтр:
# якщо P(non-zero) нижче цього порогу — ставимо 0
MIN_NON_ZERO_PROBA = 0.35

# Якщо True — expected value рахується тільки по non-zero buckets
# Якщо False — zero bucket також входить у weighted sum, але його representative value = 0
EXPECTED_ONLY_NON_ZERO = True


# ============================================================
# INPUT
# ============================================================

# Очікується, що в тебе вже є:
# X — препроцеснутий dataframe фіч
# y — таргет
#
# Наприклад:
# X = df[final_features].copy()
# y = df[TARGET_NAME].copy()

X = X.copy()
y = pd.Series(y).copy()

y_clean = pd.Series(
    np.clip(y.values, a_min=0, a_max=None),
    index=y.index,
    name=TARGET_NAME
)


# ============================================================
# TARGET BUCKETS
# ============================================================

def make_target_buckets(y_values, bins, labels):
    """
    Перетворює числовий таргет у bucket id.
    """
    bucket_names = pd.cut(
        y_values,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    bucket_id = bucket_names.cat.codes

    return bucket_id.astype(int), bucket_names.astype(str)


bucket_id, bucket_name = make_target_buckets(
    y_clean,
    bins=BUCKET_BINS,
    labels=BUCKET_LABELS
)

print("=" * 80)
print("TARGET BUCKET DISTRIBUTION")
print("=" * 80)

bucket_distribution = pd.DataFrame({
    "bucket_id": bucket_id,
    "bucket_name": bucket_name,
    "target": y_clean
}).groupby(["bucket_id", "bucket_name"]).agg(
    n=("target", "size"),
    share=("target", lambda x: len(x) / len(y_clean)),
    min_target=("target", "min"),
    median_target=("target", "median"),
    mean_target=("target", "mean"),
    max_target=("target", "max")
).reset_index()

display(bucket_distribution)


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

X_temp, X_test, y_temp, y_test, bucket_temp, bucket_test = train_test_split(
    X,
    y_clean,
    bucket_id,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=bucket_id
)

relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

X_train, X_val, y_train, y_val, bucket_train, bucket_val = train_test_split(
    X_temp,
    y_temp,
    bucket_temp,
    test_size=relative_val_size,
    random_state=RANDOM_STATE,
    stratify=bucket_temp
)

print("\n" + "=" * 80)
print("SPLIT SIZES")
print("=" * 80)
print(f"Train: {len(X_train):,}")
print(f"Val  : {len(X_val):,}")
print(f"Test : {len(X_test):,}")


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

cat_cols = [
    c for c in X_train.columns
    if X_train[c].dtype.name in ("object", "category")
]

print("\nCategorical columns:")
print(cat_cols)

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    train_categories = X_train[c].cat.categories

    X_val[c] = pd.Categorical(X_val[c], categories=train_categories)
    X_test[c] = pd.Categorical(X_test[c], categories=train_categories)


# ============================================================
# REPRESENTATIVE VALUE FOR EACH BUCKET
# ============================================================

def calculate_bucket_values(y_values, bucket_values, labels):
    """
    Для кожного bucket рахуємо representative value.
    Краще брати median, а не mean, бо таргет важкохвостий.
    Для zero bucket ставимо 0.
    """
    df_tmp = pd.DataFrame({
        "y": y_values,
        "bucket": bucket_values
    })

    result = {}

    for bucket in sorted(df_tmp["bucket"].unique()):
        part = df_tmp[df_tmp["bucket"] == bucket]["y"]

        if bucket == ZERO_BUCKET_ID:
            result[bucket] = 0.0
        else:
            result[bucket] = float(part.median())

    return result


bucket_value_map = calculate_bucket_values(
    y_values=y_train,
    bucket_values=bucket_train,
    labels=BUCKET_LABELS
)

print("\n" + "=" * 80)
print("BUCKET REPRESENTATIVE VALUES")
print("=" * 80)

for k, v in bucket_value_map.items():
    print(f"{k} | {BUCKET_LABELS[k]:<12} | value = {v:,.2f}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

def make_class_weights(y_class):
    """
    Робить ваги класів для multiclass.
    Меншим класам дає більшу вагу.
    """
    counts = pd.Series(y_class).value_counts().sort_index()
    total = counts.sum()
    n_classes = len(counts)

    class_weight = {
        int(cls): total / (n_classes * count)
        for cls, count in counts.items()
    }

    return class_weight


class_weight_map = make_class_weights(bucket_train)

print("\n" + "=" * 80)
print("CLASS WEIGHTS")
print("=" * 80)

for k, v in class_weight_map.items():
    print(f"{k} | {BUCKET_LABELS[k]:<12} | weight = {v:.4f}")

sample_weight_train = pd.Series(bucket_train).map(class_weight_map).values


# ============================================================
# MULTICLASS MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAIN MULTICLASS BUCKET MODEL")
print("=" * 80)

num_classes = len(BUCKET_LABELS)

bucket_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=num_classes,

    n_estimators=3000,
    learning_rate=0.02,

    num_leaves=31,
    max_depth=5,

    min_child_samples=30,
    subsample=0.85,
    colsample_bytree=0.85,

    reg_alpha=0.5,
    reg_lambda=1.0,

    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

bucket_model.fit(
    X_train,
    bucket_train,
    sample_weight=sample_weight_train,
    eval_set=[(X_val, bucket_val)],
    eval_metric="multi_logloss",
    categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_bucket_expected_value(
    model,
    X_data,
    bucket_value_map,
    zero_bucket_id=0,
    min_non_zero_proba=0.35,
    zero_if_zero_bucket_is_top=True,
    expected_only_non_zero=True
):
    """
    Повертає:
    - probabilities по bucket-ах
    - predicted bucket
    - expected value prediction
    - final prediction з zero-gate
    """

    proba = model.predict_proba(X_data)

    pred_bucket = np.argmax(proba, axis=1)

    bucket_values = np.array([
        bucket_value_map[i] for i in range(proba.shape[1])
    ])

    zero_proba = proba[:, zero_bucket_id]
    non_zero_proba = 1.0 - zero_proba

    if expected_only_non_zero:
        proba_non_zero_buckets = proba.copy()
        proba_non_zero_buckets[:, zero_bucket_id] = 0.0

        denom = proba_non_zero_buckets.sum(axis=1)
        denom = np.maximum(denom, 1e-12)

        normalized_non_zero_proba = proba_non_zero_buckets / denom[:, None]

        expected_value_non_zero = normalized_non_zero_proba @ bucket_values

        expected_value = non_zero_proba * expected_value_non_zero

    else:
        expected_value = proba @ bucket_values

    final_pred = expected_value.copy()

    if zero_if_zero_bucket_is_top:
        final_pred = np.where(
            pred_bucket == zero_bucket_id,
            0.0,
            final_pred
        )

    final_pred = np.where(
        non_zero_proba < min_non_zero_proba,
        0.0,
        final_pred
    )

    final_pred = np.clip(final_pred, a_min=0, a_max=None)

    return {
        "proba": proba,
        "pred_bucket": pred_bucket,
        "zero_proba": zero_proba,
        "non_zero_proba": non_zero_proba,
        "expected_value": expected_value,
        "final_pred": final_pred
    }


# ============================================================
# PREDICT TRAIN / VAL / TEST
# ============================================================

train_pred_dict = predict_bucket_expected_value(
    model=bucket_model,
    X_data=X_train,
    bucket_value_map=bucket_value_map,
    zero_bucket_id=ZERO_BUCKET_ID,
    min_non_zero_proba=MIN_NON_ZERO_PROBA,
    zero_if_zero_bucket_is_top=ZERO_IF_ZERO_BUCKET_IS_TOP,
    expected_only_non_zero=EXPECTED_ONLY_NON_ZERO
)

val_pred_dict = predict_bucket_expected_value(
    model=bucket_model,
    X_data=X_val,
    bucket_value_map=bucket_value_map,
    zero_bucket_id=ZERO_BUCKET_ID,
    min_non_zero_proba=MIN_NON_ZERO_PROBA,
    zero_if_zero_bucket_is_top=ZERO_IF_ZERO_BUCKET_IS_TOP,
    expected_only_non_zero=EXPECTED_ONLY_NON_ZERO
)

test_pred_dict = predict_bucket_expected_value(
    model=bucket_model,
    X_data=X_test,
    bucket_value_map=bucket_value_map,
    zero_bucket_id=ZERO_BUCKET_ID,
    min_non_zero_proba=MIN_NON_ZERO_PROBA,
    zero_if_zero_bucket_is_top=ZERO_IF_ZERO_BUCKET_IS_TOP,
    expected_only_non_zero=EXPECTED_ONLY_NON_ZERO
)

train_pred = train_pred_dict["final_pred"]
val_pred = val_pred_dict["final_pred"]
test_pred = test_pred_dict["final_pred"]

train_pred_bucket = train_pred_dict["pred_bucket"]
val_pred_bucket = val_pred_dict["pred_bucket"]
test_pred_bucket = test_pred_dict["pred_bucket"]


# ============================================================
# METRICS
# ============================================================

def regression_metrics(y_true, y_pred, name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_log = np.log1p(np.clip(y_true, 0, None))
    y_pred_log = np.log1p(np.clip(y_pred, 0, None))

    return {
        "dataset": name,

        "MAE_original": mean_absolute_error(y_true, y_pred),
        "MedAE_original": median_absolute_error(y_true, y_pred),
        "R2_original": r2_score(y_true, y_pred),

        "MAE_log": mean_absolute_error(y_true_log, y_pred_log),
        "MedAE_log": median_absolute_error(y_true_log, y_pred_log),
        "R2_log": r2_score(y_true_log, y_pred_log),

        "true_mean": np.mean(y_true),
        "pred_mean": np.mean(y_pred),

        "true_median": np.median(y_true),
        "pred_median": np.median(y_pred),

        "true_zero_or_low_rate": np.mean(y_true < BUCKET_BINS[1]),
        "pred_zero_rate": np.mean(y_pred == 0),

        "pred_below_100_rate": np.mean(y_pred < BUCKET_BINS[1])
    }


metrics_df = pd.DataFrame([
    regression_metrics(y_train, train_pred, "train"),
    regression_metrics(y_val, val_pred, "val"),
    regression_metrics(y_test, test_pred, "test")
])

print("\n" + "=" * 80)
print("REGRESSION-LIKE METRICS")
print("=" * 80)
display(metrics_df)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def print_bucket_classification_report(y_true_bucket, y_pred_bucket, name):
    print("\n" + "=" * 80)
    print(f"BUCKET CLASSIFICATION REPORT: {name}")
    print("=" * 80)

    print(f"Accuracy          : {accuracy_score(y_true_bucket, y_pred_bucket):.5f}")
    print(f"Balanced Accuracy : {balanced_accuracy_score(y_true_bucket, y_pred_bucket):.5f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true_bucket,
            y_pred_bucket,
            target_names=BUCKET_LABELS,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_true_bucket, y_pred_bucket)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in BUCKET_LABELS],
        columns=[f"pred_{x}" for x in BUCKET_LABELS]
    )

    display(cm_df)


print_bucket_classification_report(bucket_val, val_pred_bucket, "VALIDATION")
print_bucket_classification_report(bucket_test, test_pred_bucket, "TEST")


# ============================================================
# BUCKET DIAGNOSTICS
# ============================================================

def bucket_diagnostics(y_true, y_pred, true_bucket, pred_bucket, name):
    df_eval = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "true_bucket": np.asarray(true_bucket),
        "pred_bucket": np.asarray(pred_bucket)
    })

    df_eval["true_bucket_name"] = df_eval["true_bucket"].map(
        {i: label for i, label in enumerate(BUCKET_LABELS)}
    )

    rows = []

    for bucket_id_, part in df_eval.groupby("true_bucket"):
        rows.append({
            "dataset": name,
            "true_bucket_id": bucket_id_,
            "true_bucket_name": BUCKET_LABELS[bucket_id_],
            "n": len(part),

            "true_mean": part["y_true"].mean(),
            "pred_mean": part["y_pred"].mean(),

            "true_median": part["y_true"].median(),
            "pred_median": part["y_pred"].median(),

            "MAE": mean_absolute_error(part["y_true"], part["y_pred"]),
            "MedAE": median_absolute_error(part["y_true"], part["y_pred"]),

            "bias_pred_minus_true": part["y_pred"].mean() - part["y_true"].mean(),

            "zero_pred_rate": np.mean(part["y_pred"] == 0),
            "same_bucket_rate": np.mean(part["true_bucket"] == part["pred_bucket"])
        })

    return pd.DataFrame(rows).sort_values("true_bucket_id")


val_bucket_diag = bucket_diagnostics(
    y_true=y_val,
    y_pred=val_pred,
    true_bucket=bucket_val,
    pred_bucket=val_pred_bucket,
    name="val"
)

test_bucket_diag = bucket_diagnostics(
    y_true=y_test,
    y_pred=test_pred,
    true_bucket=bucket_test,
    pred_bucket=test_pred_bucket,
    name="test"
)

print("\n" + "=" * 80)
print("VALIDATION BUCKET DIAGNOSTICS")
print("=" * 80)
display(val_bucket_diag)

print("\n" + "=" * 80)
print("TEST BUCKET DIAGNOSTICS")
print("=" * 80)
display(test_bucket_diag)


# ============================================================
# ZERO / NON-ZERO CHECK
# ============================================================

def zero_nonzero_check(y_true, y_pred, pred_dict, name):
    true_non_zero = np.asarray(y_true) >= BUCKET_BINS[1]
    pred_non_zero = np.asarray(y_pred) > 0

    return {
        "dataset": name,

        "true_non_zero_rate": true_non_zero.mean(),
        "pred_non_zero_rate": pred_non_zero.mean(),

        "true_zero_rate": 1 - true_non_zero.mean(),
        "pred_zero_rate": 1 - pred_non_zero.mean(),

        "avg_zero_proba": pred_dict["zero_proba"].mean(),
        "avg_non_zero_proba": pred_dict["non_zero_proba"].mean()
    }


zero_check_df = pd.DataFrame([
    zero_nonzero_check(y_train, train_pred, train_pred_dict, "train"),
    zero_nonzero_check(y_val, val_pred, val_pred_dict, "val"),
    zero_nonzero_check(y_test, test_pred, test_pred_dict, "test")
])

print("\n" + "=" * 80)
print("ZERO / NON-ZERO CHECK")
print("=" * 80)
display(zero_check_df)


# ============================================================
# VALIDATION RESULT
# ============================================================

def build_result_df(X_data, y_true, true_bucket, pred_dict, dataset_name):
    result = X_data.copy()

    proba = pred_dict["proba"]

    result[TARGET_NAME + "_TRUE"] = y_true.values
    result[TARGET_NAME + "_PRED"] = pred_dict["final_pred"]
    result[TARGET_NAME + "_EXPECTED_RAW"] = pred_dict["expected_value"]

    result["TRUE_BUCKET_ID"] = np.asarray(true_bucket)
    result["TRUE_BUCKET_NAME"] = result["TRUE_BUCKET_ID"].map(
        {i: label for i, label in enumerate(BUCKET_LABELS)}
    )

    result["PRED_BUCKET_ID"] = pred_dict["pred_bucket"]
    result["PRED_BUCKET_NAME"] = result["PRED_BUCKET_ID"].map(
        {i: label for i, label in enumerate(BUCKET_LABELS)}
    )

    result["ZERO_PROBA"] = pred_dict["zero_proba"]
    result["NON_ZERO_PROBA"] = pred_dict["non_zero_proba"]

    for i, label in enumerate(BUCKET_LABELS):
        result[f"PROBA_{label}"] = proba[:, i]

    result["ABS_ERROR"] = np.abs(
        result[TARGET_NAME + "_TRUE"] - result[TARGET_NAME + "_PRED"]
    )

    result["LOG_ABS_ERROR"] = np.abs(
        np.log1p(result[TARGET_NAME + "_TRUE"]) -
        np.log1p(result[TARGET_NAME + "_PRED"])
    )

    result["DATASET"] = dataset_name

    return result


validation_result = build_result_df(
    X_data=X_val,
    y_true=y_val,
    true_bucket=bucket_val,
    pred_dict=val_pred_dict,
    dataset_name="val"
)

test_result = build_result_df(
    X_data=X_test,
    y_true=y_test,
    true_bucket=bucket_test,
    pred_dict=test_pred_dict,
    dataset_name="test"
)

print("\nvalidation_result shape:", validation_result.shape)
display(validation_result.head())

print("\ntest_result shape:", test_result.shape)
display(test_result.head())


# ============================================================
# DISTRIBUTION PLOTS
# ============================================================

plt.figure(figsize=(10, 6))
sns.kdeplot(np.log1p(y_test), label="True", linewidth=2)
sns.kdeplot(np.log1p(test_pred), label="Predicted", linewidth=2, linestyle="--")
plt.title("Test Distribution Match: log1p")
plt.xlabel(f"log1p({TARGET_NAME})")
plt.legend()
plt.show()


plt.figure(figsize=(8, 8))
plt.scatter(
    np.log1p(y_test),
    np.log1p(test_pred),
    alpha=0.25,
    s=15
)

mx = max(
    float(np.log1p(y_test).max()),
    float(np.log1p(test_pred).max())
)

plt.plot([0, mx], [0, mx], "--", linewidth=2)
plt.title("Test: True vs Predicted, log1p")
plt.xlabel(f"True {TARGET_NAME}, log1p")
plt.ylabel(f"Predicted {TARGET_NAME}, log1p")
plt.show()


# ============================================================
# BUCKET MEAN COMPARISON PLOT
# ============================================================

plot_df = test_bucket_diag.copy()

plt.figure(figsize=(12, 6))
x = np.arange(len(plot_df))

plt.bar(x - 0.2, plot_df["true_mean"], width=0.4, label="True mean")
plt.bar(x + 0.2, plot_df["pred_mean"], width=0.4, label="Pred mean")

plt.xticks(x, plot_df["true_bucket_name"], rotation=30, ha="right")
plt.title("Test: True Mean vs Predicted Mean by True Bucket")
plt.ylabel(TARGET_NAME)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": bucket_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n" + "=" * 80)
print("TOP FEATURES")
print("=" * 80)

display(feature_importance.head(40))