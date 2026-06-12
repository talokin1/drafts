import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    mean_absolute_error,
    median_absolute_error,
    r2_score
)
from sklearn.isotonic import IsotonicRegression


RANDOM_STATE = 42

TARGET_NAME = "FX"    
ACTIVE_THRESHOLD_TARGET = 100
clf_features = [c for c in final_features if c != "FX_TYPE"]

# Для регресії FX_TYPE залишаємо
reg_features = final_features.copy()

print("Classification features:", len(clf_features))
print("Regression features:", len(reg_features))
print("FX_TYPE in clf:", "FX_TYPE" in clf_features)
print("FX_TYPE in reg:", "FX_TYPE" in reg_features)

df_train = df_train.copy()
df_val = df_val.copy()

df_train[TARGET_NAME] = pd.to_numeric(df_train[TARGET_NAME], errors="coerce").fillna(0)
df_val[TARGET_NAME] = pd.to_numeric(df_val[TARGET_NAME], errors="coerce").fillna(0)

df_train["FX_ACTIVE"] = (df_train[TARGET_NAME] > ACTIVE_THRESHOLD_TARGET).astype(int)
df_val["FX_ACTIVE"] = (df_val[TARGET_NAME] > ACTIVE_THRESHOLD_TARGET).astype(int)

print("Train rows:", len(df_train))
print("Val rows:", len(df_val))
print("Train active rate:", round(df_train["FX_ACTIVE"].mean(), 4))
print("Val active rate:", round(df_val["FX_ACTIVE"].mean(), 4))

df_clf_fit, df_clf_calib = train_test_split(
    df_train,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=df_train["FX_ACTIVE"]
)

print("\nClassifier fit rows:", df_clf_fit.shape[0])
print("Calibration rows:", df_clf_calib.shape[0])
print("Fit active rate:", round(df_clf_fit["FX_ACTIVE"].mean(), 4))
print("Calib active rate:", round(df_clf_calib["FX_ACTIVE"].mean(), 4))


X_clf_fit, X_clf_calib, cat_cols_clf = prepare_train_val_X(
    df_train=df_clf_fit,
    df_val=df_clf_calib,
    features=clf_features
)

y_clf_fit = df_clf_fit["FX_ACTIVE"]
y_clf_calib = df_clf_calib["FX_ACTIVE"]

print("\nClassification X fit shape:", X_clf_fit.shape)
print("Classification X calib shape:", X_clf_calib.shape)
print("Categorical columns clf:", cat_cols_clf)

bad_cols_clf = X_clf_fit.select_dtypes(include=["object"]).columns.tolist()
print("Object columns clf after preprocessing:", bad_cols_clf)










# ---------Clasidfcation
n_pos = y_clf_fit.sum()
n_neg = len(y_clf_fit) - n_pos

scale_pos_weight = np.sqrt(n_neg / max(n_pos, 1))

print("\nn_pos:", int(n_pos))
print("n_neg:", int(n_neg))
print("scale_pos_weight:", round(scale_pos_weight, 3))

clf_binary = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=3000,
    learning_rate=0.02,

    # трохи менш задушена модель, ніж попередня
    num_leaves=15,
    max_depth=4,

    min_child_samples=150,
    min_child_weight=1e-2,
    min_split_gain=0.01,

    reg_alpha=2.0,
    reg_lambda=8.0,

    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.75,

    cat_smooth=30,
    cat_l2=20,
    min_data_per_group=100,
    max_cat_threshold=16,

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

clf_binary.fit(
    X_clf_fit,
    y_clf_fit,
    eval_set=[(X_clf_calib, y_clf_calib)],
    eval_metric="binary_logloss",
    categorical_feature=cat_cols_clf,
    callbacks=[
        lgb.early_stopping(150),
        lgb.log_evaluation(100)
    ]
)

probs_calib_raw = clf_binary.predict_proba(X_clf_calib)[:, 1]

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(probs_calib_raw, y_clf_calib)

probs_calib = calibrator.predict(probs_calib_raw)

print("\nRaw probability quantiles on calibration:")
print(pd.Series(probs_calib_raw).quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\nCalibrated probability quantiles on calibration:")
print(pd.Series(probs_calib).quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))


precision, recall, thresholds = precision_recall_curve(y_clf_calib, probs_calib)

# thresholds на 1 коротший за precision/recall
f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("\nBest threshold by F1 on calibration:", round(float(best_threshold), 5))
print("Best F1:", round(float(f1_scores[best_idx]), 5))
print("Precision:", round(float(precision[best_idx]), 5))
print("Recall:", round(float(recall[best_idx]), 5))

CLASSIFICATION_THRESHOLD = float(best_threshold)

_, X_val_clf, _ = prepare_train_val_X(
    df_train=df_clf_fit,
    df_val=df_val,
    features=clf_features
)

y_val_clf = df_val["FX_ACTIVE"]

probs_val_raw = clf_binary.predict_proba(X_val_clf)[:, 1]
p_fx_val = calibrator.predict(probs_val_raw)

preds_val_clf = (p_fx_val >= CLASSIFICATION_THRESHOLD).astype(int)

roc_auc = roc_auc_score(y_val_clf, p_fx_val)
pr_auc = average_precision_score(y_val_clf, p_fx_val)

print("\n" + "=" * 70)
print("CLASSIFICATION VALIDATION")
print("=" * 70)
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC :", round(pr_auc, 4))
print("Threshold:", round(CLASSIFICATION_THRESHOLD, 5))

print("\nClassification report:")
print(classification_report(y_val_clf, preds_val_clf))

print("Confusion matrix:")
print(confusion_matrix(y_val_clf, preds_val_clf))

print("\nP_FX validation quantiles:")
print(pd.Series(p_fx_val).quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))

print("\nPredicted active rate on validation:", round(preds_val_clf.mean(), 4))
print("True active rate on validation     :", round(y_val_clf.mean(), 4))












# REGRESSION DATA: ONLY ACTIVE CLIENTS
df_train_reg = df_train[df_train[TARGET_NAME] > ACTIVE_THRESHOLD_TARGET].copy()
df_val_reg = df_val[df_val[TARGET_NAME] > ACTIVE_THRESHOLD_TARGET].copy()

print("\nTrain regression rows:", df_train_reg.shape[0])
print("Val regression rows:", df_val_reg.shape[0])

FX_UPPER_CAP = df_train_reg[TARGET_NAME].quantile(0.999)

print("FX upper cap:", FX_UPPER_CAP)

df_train_reg["FX_CAPPED"] = df_train_reg[TARGET_NAME].clip(upper=FX_UPPER_CAP)
df_val_reg["FX_CAPPED"] = df_val_reg[TARGET_NAME].clip(upper=FX_UPPER_CAP)

df_train_reg["FX_LOG"] = np.log1p(df_train_reg["FX_CAPPED"])
df_val_reg["FX_LOG"] = np.log1p(df_val_reg["FX_CAPPED"])


X_train_reg, X_val_reg, cat_cols_reg = prepare_train_val_X(
    df_train=df_train_reg,
    df_val=df_val_reg,
    features=reg_features
)

y_train_reg = df_train_reg["FX_LOG"]
y_val_reg = df_val_reg["FX_LOG"]

print("\nRegression X train shape:", X_train_reg.shape)
print("Regression X val shape:", X_val_reg.shape)
print("Categorical columns reg:", cat_cols_reg)

print("\nDtypes check:")
print(X_train_reg.dtypes.value_counts())

bad_cols_reg = X_train_reg.select_dtypes(include=["object"]).columns.tolist()
print("\nObject columns reg after preprocessing:", bad_cols_reg)


# ============================================================
# REGRESSOR ON LOG TARGET
# ============================================================

reg = lgb.LGBMRegressor(
    objective="regression_l1",

    n_estimators=5000,
    learning_rate=0.02,

    num_leaves=31,
    max_depth=5,
    min_child_samples=40,

    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,

    reg_alpha=1.0,
    reg_lambda=3.0,

    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

reg.fit(
    X_train_reg,
    y_train_reg,
    eval_set=[(X_val_reg, y_val_reg)],
    eval_metric="l1",
    categorical_feature=cat_cols_reg,
    callbacks=[
        lgb.early_stopping(200),
        lgb.log_evaluation(100)
    ]
)


# ============================================================
# REGRESSION METRICS ON ACTIVE VALIDATION CLIENTS
# ============================================================

pred_log_val_reg = reg.predict(X_val_reg)
pred_log_val_reg = np.clip(pred_log_val_reg, 0, None)

y_true_log = y_val_reg.values
y_pred_log = pred_log_val_reg

y_true_reg = np.expm1(y_true_log)
y_pred_reg = np.expm1(y_pred_log)

mae_log = mean_absolute_error(y_true_log, y_pred_log)
medae_log = median_absolute_error(y_true_log, y_pred_log)
r2_log = r2_score(y_true_log, y_pred_log)

mae_orig = mean_absolute_error(y_true_reg, y_pred_reg)
medae_orig = median_absolute_error(y_true_reg, y_pred_reg)

eps = 1e-9
mape = np.mean(np.abs(y_true_reg - y_pred_reg) / np.maximum(np.abs(y_true_reg), eps))
smape = np.mean(
    2.0 * np.abs(y_true_reg - y_pred_reg) /
    np.maximum(np.abs(y_true_reg) + np.abs(y_pred_reg), eps)
)

print("\n" + "=" * 70)
print("REGRESSION METRICS ON ACTIVE CLIENTS")
print("=" * 70)
print("MAE_log       :", round(mae_log, 5))
print("MedAE_log    :", round(medae_log, 5))
print("R2_log       :", round(r2_log, 5))
print("-" * 70)
print("MAE original :", round(mae_orig, 2))
print("MedAE original:", round(medae_orig, 2))
print("MAPE         :", round(mape, 4))
print("SMAPE        :", round(smape, 4))

print("\nActive validation true/pred max:")
print("VAL true max:", y_true_reg.max())
print("VAL pred max:", y_pred_reg.max())

print("\nActive validation pred quantiles:")
print(pd.Series(y_pred_reg).quantile([0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))




















# ------- FINAL
_, X_val_all_reg, _ = prepare_train_val_X(
    df_train=df_train_reg,
    df_val=df_val,
    features=reg_features
)

pred_log_cond_val = reg.predict(X_val_all_reg)
pred_log_cond_val = np.clip(pred_log_cond_val, 0, None)

pred_amount_cond_val = np.expm1(pred_log_cond_val)

# Обмежимо регресійний прогноз верхнім cap, щоб не вилітав вище навчального діапазону
pred_amount_cond_val = np.clip(pred_amount_cond_val, 0, FX_UPPER_CAP)

# Soft expected value — лише для аналізу
fx_expected_soft_val = p_fx_val * pred_amount_cond_val

# Hard final logic:
# якщо класифікатор сказав inactive -> 0
# якщо active -> прогноз регресії
fx_final_hard_val = np.where(
    preds_val_clf == 1,
    pred_amount_cond_val,
    0.0
)

validation_results = df_val[[TARGET_NAME]].copy()

validation_results["P_FX_RAW"] = probs_val_raw
validation_results["P_FX"] = p_fx_val
validation_results["FX_ACTIVE_TRUE"] = df_val["FX_ACTIVE"].values
validation_results["FX_ACTIVE_PRED"] = preds_val_clf

validation_results["FX_COND_PRED"] = pred_amount_cond_val
validation_results["FX_EXPECTED_SOFT"] = fx_expected_soft_val
validation_results["FX_FINAL_PRED"] = fx_final_hard_val

validation_results = validation_results.rename(columns={
    TARGET_NAME: "FX_TRUE"
})

validation_results.head()





# ============================================================
# FINAL METRICS ON ALL VALIDATION CLIENTS
# ============================================================

y_true_all = validation_results["FX_TRUE"].values
y_pred_all = validation_results["FX_FINAL_PRED"].values
y_pred_soft = validation_results["FX_EXPECTED_SOFT"].values

mae_all = mean_absolute_error(y_true_all, y_pred_all)
medae_all = median_absolute_error(y_true_all, y_pred_all)

mae_soft = mean_absolute_error(y_true_all, y_pred_soft)
medae_soft = median_absolute_error(y_true_all, y_pred_soft)

true_sum = y_true_all.sum()
pred_sum = y_pred_all.sum()
pred_soft_sum = y_pred_soft.sum()

pred_true_ratio = pred_sum / max(true_sum, 1)
pred_soft_true_ratio = pred_soft_sum / max(true_sum, 1)

eps = 1e-9
smape_all = np.mean(
    2.0 * np.abs(y_true_all - y_pred_all) /
    np.maximum(np.abs(y_true_all) + np.abs(y_pred_all), eps)
)

print("=" * 70)
print("FINAL FX MODEL — VALIDATION")
print("=" * 70)

print("Rows:", len(validation_results))
print("True active clients:", int((validation_results["FX_TRUE"] > ACTIVE_THRESHOLD_TARGET).sum()))
print("Pred active clients:", int(validation_results["FX_ACTIVE_PRED"].sum()))

print("-" * 70)
print("HARD FINAL PREDICTION")
print("MAE:", round(mae_all, 2))
print("MedAE:", round(medae_all, 2))
print("SMAPE:", round(smape_all, 4))

print("-" * 70)
print("SUM CHECK")
print("True sum:", round(true_sum, 2))
print("Pred hard sum:", round(pred_sum, 2))
print("Pred hard / true:", round(pred_true_ratio, 4))

print("-" * 70)
print("SOFT EXPECTED VALUE CHECK")
print("Soft MAE:", round(mae_soft, 2))
print("Soft MedAE:", round(medae_soft, 2))
print("Soft pred sum:", round(pred_soft_sum, 2))
print("Soft pred / true:", round(pred_soft_true_ratio, 4))

print("-" * 70)
print("FX_FINAL_PRED quantiles:")
print(validation_results["FX_FINAL_PRED"].quantile([0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]))

print("-" * 70)
print("FX_COND_PRED quantiles:")
print(validation_results["FX_COND_PRED"].quantile([0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]))

print("-" * 70)
print("P_FX quantiles:")
print(validation_results["P_FX"].quantile([0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]))