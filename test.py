import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

TARGET_NAME = "CURR_ACC"   # зміни, якщо інший таргет
THRESHOLD = 100            # з якого доходу клієнт вважається активним / прибутковим

TEST_SIZE = 0.20
VAL_SIZE = 0.20

N_STRATA_BUCKETS = 5

MIN_PROBA_TO_FORCE_ZERO = 0.08
USE_SOFT_GATE = True       # True: proba * reg_pred; False: hard gate

sns.set_theme(style="whitegrid")


# ============================================================
# BASIC INPUT CHECK
# ============================================================

# Очікується, що в тебе вже є:
# X — препроцеснутий датафрейм фіч
# y — таргет
#
# Наприклад:
# X = df[final_features]
# y = df[TARGET_NAME]

X = X.copy()
y = pd.Series(y).copy()

# прибираємо негативні значення таргету
y_clean = pd.Series(
    np.clip(y.values, a_min=0, a_max=None),
    index=y.index,
    name=TARGET_NAME
)

# бінарний таргет: клієнт генерує дохід чи ні
y_binary = (y_clean >= THRESHOLD).astype(int)


# ============================================================
# STRATIFICATION BY TARGET BUCKETS
# ============================================================

def make_target_strata(y_values, threshold=100, n_bins=5):
    """
    Робить strata для train/val/test split:
    - zero_or_low: клієнти нижче threshold
    - positive buckets: квантильні групи серед активних клієнтів
    """
    y_values = pd.Series(y_values).copy()
    strata = pd.Series("zero_or_low", index=y_values.index, dtype="object")

    positive_mask = y_values >= threshold

    if positive_mask.sum() > n_bins:
        positive_buckets = pd.qcut(
            y_values.loc[positive_mask],
            q=n_bins,
            labels=[f"positive_q{i+1}" for i in range(n_bins)],
            duplicates="drop"
        ).astype(str)

        strata.loc[positive_mask] = positive_buckets
    else:
        strata.loc[positive_mask] = "positive"

    return strata.astype(str)


strata = make_target_strata(
    y_clean,
    threshold=THRESHOLD,
    n_bins=N_STRATA_BUCKETS
)

print("Target strata distribution:")
print(strata.value_counts(normalize=True).sort_index())


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

# 1) Спочатку відділяємо test
X_temp, X_test, y_temp, y_test, strata_temp, strata_test = train_test_split(
    X,
    y_clean,
    strata,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=strata
)

# 2) Потім із temp відділяємо validation
# Якщо test = 20%, val = 20%, то val всередині temp = 20 / 80 = 0.25
relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

X_train, X_val, y_train, y_val, strata_train, strata_val = train_test_split(
    X_temp,
    y_temp,
    strata_temp,
    test_size=relative_val_size,
    random_state=RANDOM_STATE,
    stratify=strata_temp
)

print("\nSplit sizes:")
print(f"Train: {X_train.shape[0]:,}")
print(f"Val  : {X_val.shape[0]:,}")
print(f"Test : {X_test.shape[0]:,}")


# ============================================================
# CATEGORICAL COLUMNS HANDLING
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

    # фіксуємо категорії train, щоб val/test мали ту саму структуру
    train_categories = X_train[c].cat.categories

    X_val[c] = pd.Categorical(X_val[c], categories=train_categories)
    X_test[c] = pd.Categorical(X_test[c], categories=train_categories)


# ============================================================
# STAGE 1: CLASSIFIER ZERO / NON-ZERO
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1: CLASSIFIER")
print("=" * 70)

y_train_clf = (y_train >= THRESHOLD).astype(int)
y_val_clf = (y_val >= THRESHOLD).astype(int)
y_test_clf = (y_test >= THRESHOLD).astype(int)

neg_count = (y_train_clf == 0).sum()
pos_count = (y_train_clf == 1).sum()

scale_pos_weight = neg_count / max(pos_count, 1)

print(f"Train negative: {neg_count:,}")
print(f"Train positive: {pos_count:,}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train,
    y_train_clf,
    eval_set=[(X_val, y_val_clf)],
    eval_metric="auc",
    categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

val_proba = clf.predict_proba(X_val)[:, 1]
test_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassifier metrics:")
print(f"ROC-AUC Val : {roc_auc_score(y_val_clf, val_proba):.5f}")
print(f"ROC-AUC Test: {roc_auc_score(y_test_clf, test_proba):.5f}")
print(f"PR-AUC Val  : {average_precision_score(y_val_clf, val_proba):.5f}")
print(f"PR-AUC Test : {average_precision_score(y_test_clf, test_proba):.5f}")


# ============================================================
# THRESHOLD TUNING ON VALIDATION
# ============================================================

def evaluate_classification_thresholds(y_true, proba):
    """
    Перебирає thresholds і повертає таблицю precision/recall/f1.
    """
    rows = []

    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= thr).astype(int)

        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        rows.append({
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "predicted_positive_rate": pred.mean()
        })

    return pd.DataFrame(rows)


threshold_table = evaluate_classification_thresholds(y_val_clf.values, val_proba)

# Базовий варіант: максимум F1
best_row = threshold_table.loc[threshold_table["f1"].idxmax()]
CLASSIFICATION_THRESHOLD = float(best_row["threshold"])

print("\nBest classification threshold by F1 on validation:")
print(best_row)


# ============================================================
# STAGE 2: REGRESSOR ON ACTIVE CLIENTS
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: REGRESSOR")
print("=" * 70)

mask_train_reg = y_train >= THRESHOLD
mask_val_reg = y_val >= THRESHOLD
mask_test_reg = y_test >= THRESHOLD

X_train_reg = X_train.loc[mask_train_reg].copy()
X_val_reg = X_val.loc[mask_val_reg].copy()
X_test_reg = X_test.loc[mask_test_reg].copy()

y_train_reg_raw = y_train.loc[mask_train_reg].copy()
y_val_reg_raw = y_val.loc[mask_val_reg].copy()
y_test_reg_raw = y_test.loc[mask_test_reg].copy()

y_train_reg_log = np.log1p(y_train_reg_raw)
y_val_reg_log = np.log1p(y_val_reg_raw)
y_test_reg_log = np.log1p(y_test_reg_raw)

print(f"Reg train size: {X_train_reg.shape[0]:,}")
print(f"Reg val size  : {X_val_reg.shape[0]:,}")
print(f"Reg test size : {X_test_reg.shape[0]:,}")

# Ваги для хвоста
# Ідея: не видаляти багатих клієнтів, а дати їм більшу вагу.
sample_weight = pd.Series(
    np.ones(len(y_train_reg_raw)),
    index=y_train_reg_raw.index
)

q70 = y_train_reg_raw.quantile(0.70)
q85 = y_train_reg_raw.quantile(0.85)
q95 = y_train_reg_raw.quantile(0.95)

sample_weight.loc[y_train_reg_raw >= q70] = 1.5
sample_weight.loc[y_train_reg_raw >= q85] = 2.5
sample_weight.loc[y_train_reg_raw >= q95] = 4.0

print("\nTail weighting thresholds:")
print(f"q70: {q70:,.2f}")
print(f"q85: {q85:,.2f}")
print(f"q95: {q95:,.2f}")

reg = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=25,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train_reg,
    y_train_reg_log,
    sample_weight=sample_weight.loc[X_train_reg.index],
    eval_set=[(X_val_reg, y_val_reg_log)],
    eval_metric="mae",
    categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
    callbacks=[
        lgb.early_stopping(stopping_rounds=250, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_two_stage(
    X_data,
    clf_model,
    reg_model,
    classification_threshold=0.5,
    min_proba_to_force_zero=0.08,
    use_soft_gate=True
):
    """
    Two-stage prediction.

    use_soft_gate=True:
        final_pred = proba_non_zero * reg_pred
        Але якщо proba дуже мала — ставимо 0.

    use_soft_gate=False:
        final_pred = reg_pred тільки якщо proba >= threshold, інакше 0.
    """

    proba_non_zero = clf_model.predict_proba(X_data)[:, 1]

    reg_pred_log = reg_model.predict(X_data)
    reg_pred = np.expm1(reg_pred_log)
    reg_pred = np.clip(reg_pred, a_min=0, a_max=None)

    if use_soft_gate:
        final_pred = proba_non_zero * reg_pred
        final_pred = np.where(
            proba_non_zero < min_proba_to_force_zero,
            0,
            final_pred
        )
    else:
        class_pred = (proba_non_zero >= classification_threshold).astype(int)
        final_pred = np.where(class_pred == 1, reg_pred, 0)

    return {
        "proba_non_zero": proba_non_zero,
        "reg_pred": reg_pred,
        "final_pred": final_pred
    }


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_pred_dict = predict_two_stage(
    X_val,
    clf,
    reg,
    classification_threshold=CLASSIFICATION_THRESHOLD,
    min_proba_to_force_zero=MIN_PROBA_TO_FORCE_ZERO,
    use_soft_gate=USE_SOFT_GATE
)

val_final_pred = val_pred_dict["final_pred"]
val_reg_pred_all = val_pred_dict["reg_pred"]
val_proba = val_pred_dict["proba_non_zero"]


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_pred_dict = predict_two_stage(
    X_test,
    clf,
    reg,
    classification_threshold=CLASSIFICATION_THRESHOLD,
    min_proba_to_force_zero=MIN_PROBA_TO_FORCE_ZERO,
    use_soft_gate=USE_SOFT_GATE
)

test_final_pred = test_pred_dict["final_pred"]
test_reg_pred_all = test_pred_dict["reg_pred"]
test_proba = test_pred_dict["proba_non_zero"]


# ============================================================
# METRICS FUNCTIONS
# ============================================================

def regression_metrics(y_true, y_pred, name="dataset"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_log = np.log1p(np.clip(y_true, 0, None))
    y_pred_log = np.log1p(np.clip(y_pred, 0, None))

    metrics = {
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

        "true_zero_rate": np.mean(y_true < THRESHOLD),
        "pred_zero_rate": np.mean(y_pred < THRESHOLD),
    }

    return metrics


def bucket_metrics(y_true, y_pred, threshold=100):
    """
    Дивимось якість окремо по групах:
    - zero_or_low
    - low_positive
    - middle_positive
    - high_positive
    - very_high_positive
    """

    df_eval = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred)
    })

    active = df_eval["y_true"] >= threshold

    df_eval["bucket"] = "zero_or_low"

    if active.sum() > 10:
        q1 = df_eval.loc[active, "y_true"].quantile(0.25)
        q2 = df_eval.loc[active, "y_true"].quantile(0.50)
        q3 = df_eval.loc[active, "y_true"].quantile(0.75)
        q9 = df_eval.loc[active, "y_true"].quantile(0.90)

        df_eval.loc[(active) & (df_eval["y_true"] < q1), "bucket"] = "low_positive"
        df_eval.loc[(active) & (df_eval["y_true"] >= q1) & (df_eval["y_true"] < q2), "bucket"] = "mid_low_positive"
        df_eval.loc[(active) & (df_eval["y_true"] >= q2) & (df_eval["y_true"] < q3), "bucket"] = "mid_high_positive"
        df_eval.loc[(active) & (df_eval["y_true"] >= q3) & (df_eval["y_true"] < q9), "bucket"] = "high_positive"
        df_eval.loc[(active) & (df_eval["y_true"] >= q9), "bucket"] = "very_high_positive"

    rows = []

    for bucket, part in df_eval.groupby("bucket"):
        rows.append({
            "bucket": bucket,
            "n": len(part),
            "true_mean": part["y_true"].mean(),
            "pred_mean": part["y_pred"].mean(),
            "true_median": part["y_true"].median(),
            "pred_median": part["y_pred"].median(),
            "MAE": mean_absolute_error(part["y_true"], part["y_pred"]),
            "MedAE": median_absolute_error(part["y_true"], part["y_pred"]),
            "bias_pred_minus_true": part["y_pred"].mean() - part["y_true"].mean()
        })

    return pd.DataFrame(rows).sort_values("true_mean")


# ============================================================
# FINAL METRICS
# ============================================================

train_pred_dict = predict_two_stage(
    X_train,
    clf,
    reg,
    classification_threshold=CLASSIFICATION_THRESHOLD,
    min_proba_to_force_zero=MIN_PROBA_TO_FORCE_ZERO,
    use_soft_gate=USE_SOFT_GATE
)

train_final_pred = train_pred_dict["final_pred"]

metrics_df = pd.DataFrame([
    regression_metrics(y_train, train_final_pred, name="train_combined"),
    regression_metrics(y_val, val_final_pred, name="val_combined"),
    regression_metrics(y_test, test_final_pred, name="test_combined")
])

print("\n" + "=" * 70)
print("COMBINED PIPELINE METRICS")
print("=" * 70)
display(metrics_df)


# ============================================================
# REGRESSOR ONLY METRICS ON ACTIVE CLIENTS
# ============================================================

train_reg_pred_log = reg.predict(X_train_reg)
val_reg_pred_log = reg.predict(X_val_reg)
test_reg_pred_log = reg.predict(X_test_reg)

train_reg_pred = np.expm1(train_reg_pred_log)
val_reg_pred = np.expm1(val_reg_pred_log)
test_reg_pred = np.expm1(test_reg_pred_log)

reg_only_metrics_df = pd.DataFrame([
    regression_metrics(y_train_reg_raw, train_reg_pred, name="train_reg_active_only"),
    regression_metrics(y_val_reg_raw, val_reg_pred, name="val_reg_active_only"),
    regression_metrics(y_test_reg_raw, test_reg_pred, name="test_reg_active_only")
])

print("\n" + "=" * 70)
print("REGRESSOR ONLY METRICS ON ACTIVE CLIENTS")
print("=" * 70)
display(reg_only_metrics_df)


# ============================================================
# CLASSIFICATION REPORT ON TEST
# ============================================================

test_class_pred = (test_proba >= CLASSIFICATION_THRESHOLD).astype(int)

print("\n" + "=" * 70)
print("CLASSIFIER TEST REPORT")
print("=" * 70)

print(f"Chosen threshold: {CLASSIFICATION_THRESHOLD:.4f}")
print(classification_report(y_test_clf, test_class_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test_clf, test_class_pred))


# ============================================================
# BUCKET DIAGNOSTICS
# ============================================================

val_bucket_metrics = bucket_metrics(y_val, val_final_pred, threshold=THRESHOLD)
test_bucket_metrics = bucket_metrics(y_test, test_final_pred, threshold=THRESHOLD)

print("\n" + "=" * 70)
print("VALIDATION BUCKET METRICS")
print("=" * 70)
display(val_bucket_metrics)

print("\n" + "=" * 70)
print("TEST BUCKET METRICS")
print("=" * 70)
display(test_bucket_metrics)


# ============================================================
# VALIDATION RESULT DATAFRAME
# ============================================================

validation_result = X_val.copy()
validation_result[TARGET_NAME + "_TRUE"] = y_val.values
validation_result[TARGET_NAME + "_PRED"] = val_final_pred
validation_result[TARGET_NAME + "_REG_PRED"] = val_reg_pred_all
validation_result["PROBA_NON_ZERO"] = val_proba
validation_result["TRUE_IS_ACTIVE"] = (y_val.values >= THRESHOLD).astype(int)
validation_result["PRED_IS_ACTIVE"] = (val_proba >= CLASSIFICATION_THRESHOLD).astype(int)
validation_result["ABS_ERROR"] = np.abs(validation_result[TARGET_NAME + "_TRUE"] - validation_result[TARGET_NAME + "_PRED"])
validation_result["LOG_ABS_ERROR"] = np.abs(
    np.log1p(validation_result[TARGET_NAME + "_TRUE"]) -
    np.log1p(validation_result[TARGET_NAME + "_PRED"])
)

print("\nvalidation_result shape:", validation_result.shape)
display(validation_result.head())


# ============================================================
# TEST RESULT DATAFRAME
# ============================================================

test_result = X_test.copy()
test_result[TARGET_NAME + "_TRUE"] = y_test.values
test_result[TARGET_NAME + "_PRED"] = test_final_pred
test_result[TARGET_NAME + "_REG_PRED"] = test_reg_pred_all
test_result["PROBA_NON_ZERO"] = test_proba
test_result["TRUE_IS_ACTIVE"] = (y_test.values >= THRESHOLD).astype(int)
test_result["PRED_IS_ACTIVE"] = (test_proba >= CLASSIFICATION_THRESHOLD).astype(int)
test_result["ABS_ERROR"] = np.abs(test_result[TARGET_NAME + "_TRUE"] - test_result[TARGET_NAME + "_PRED"])
test_result["LOG_ABS_ERROR"] = np.abs(
    np.log1p(test_result[TARGET_NAME + "_TRUE"]) -
    np.log1p(test_result[TARGET_NAME + "_PRED"])
)

print("\ntest_result shape:", test_result.shape)
display(test_result.head())


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter log-space
axes[0].scatter(
    np.log1p(y_test),
    np.log1p(test_final_pred),
    alpha=0.25,
    s=15
)

mn = 0
mx = max(
    float(np.log1p(y_test).max()),
    float(np.log1p(test_final_pred).max())
)

axes[0].plot([mn, mx], [mn, mx], "--", linewidth=2)
axes[0].set_title("Test: True vs Predicted, log1p")
axes[0].set_xlabel(f"True {TARGET_NAME}, log1p")
axes[0].set_ylabel(f"Predicted {TARGET_NAME}, log1p")

# Distribution
sns.kdeplot(
    np.log1p(y_test),
    ax=axes[1],
    label="True distribution",
    linewidth=2
)

sns.kdeplot(
    np.log1p(test_final_pred),
    ax=axes[1],
    label="Predicted distribution",
    linewidth=2,
    linestyle="--"
)

axes[1].set_title("Test: Distribution Match, log1p")
axes[1].set_xlabel(f"log1p({TARGET_NAME})")
axes[1].legend()

plt.tight_layout()
plt.show()


# ============================================================
# ZERO / NON-ZERO DISTRIBUTION CHECK
# ============================================================

zero_check = pd.DataFrame({
    "dataset": ["train", "val", "test"],
    "true_zero_or_low_rate": [
        np.mean(y_train < THRESHOLD),
        np.mean(y_val < THRESHOLD),
        np.mean(y_test < THRESHOLD)
    ],
    "pred_zero_or_low_rate": [
        np.mean(train_final_pred < THRESHOLD),
        np.mean(val_final_pred < THRESHOLD),
        np.mean(test_final_pred < THRESHOLD)
    ],
    "true_active_rate": [
        np.mean(y_train >= THRESHOLD),
        np.mean(y_val >= THRESHOLD),
        np.mean(y_test >= THRESHOLD)
    ],
    "pred_active_rate": [
        np.mean(train_final_pred >= THRESHOLD),
        np.mean(val_final_pred >= THRESHOLD),
        np.mean(test_final_pred >= THRESHOLD)
    ]
})

print("\n" + "=" * 70)
print("ZERO / NON-ZERO RATE CHECK")
print("=" * 70)
display(zero_check)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

clf_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

reg_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": reg.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop classifier features:")
display(clf_importance.head(30))

print("\nTop regressor features:")
display(reg_importance.head(30))