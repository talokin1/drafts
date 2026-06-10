import numpy as np
import pandas as pd
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    r2_score
)

RANDOM_STATE = 42

# =========================
# CONFIG
# =========================

TARGET_NAME = "FX"          # заміни, якщо в тебе target називається інакше
ID_COL = "IDENTIFYCODE"     # якщо IDENTIFYCODE у тебе в index, код нижче теж спрацює

# Важливо: final_features має вже існувати
# final_features = [...]


df_base = df.copy()

df_base[TARGET_NAME] = df_base[TARGET_NAME].clip(lower=0)

if ID_COL in df_base.columns:
    df_base = df_base.set_index(ID_COL, drop=False)

print("Dataset shape:", df_base.shape)
print("Target sum:", df_base[TARGET_NAME].sum())
print("Active clients:", (df_base[TARGET_NAME] > 0).sum())
print("Inactive clients:", (df_base[TARGET_NAME] == 0).sum())
print("Active share:", round((df_base[TARGET_NAME] > 0).mean(), 4))


plt.figure(figsize=(8, 4))
sns.histplot(df_base[TARGET_NAME], bins=200)
plt.title("FX target distribution")
plt.xlabel(TARGET_NAME)
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(np.log1p(df_base[TARGET_NAME]), bins=100)
plt.title("log1p(FX) target distribution")
plt.xlabel(f"log1p({TARGET_NAME})")
plt.ylabel("Count")
plt.show()

print(df_base[TARGET_NAME].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]))
















y_clf_full = (df_base[TARGET_NAME] > 0).astype(int)

train_idx, val_idx = train_test_split(
    df_base.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_clf_full
)

df_train = df_base.loc[train_idx].copy()
df_val = df_base.loc[val_idx].copy()

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)

print("\nTrain active share:", round((df_train[TARGET_NAME] > 0).mean(), 4))
print("Val active share:", round((df_val[TARGET_NAME] > 0).mean(), 4))
















# ------------- HELPERS
y_clf_full = (df_base[TARGET_NAME] > 0).astype(int)

train_idx, val_idx = train_test_split(
    df_base.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_clf_full
)

df_train = df_base.loc[train_idx].copy()
df_val = df_base.loc[val_idx].copy()

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)

print("\nTrain active share:", round((df_train[TARGET_NAME] > 0).mean(), 4))
print("Val active share:", round((df_val[TARGET_NAME] > 0).mean(), 4))





# ----------- classification



X_train_clf, cat_cols = prepare_X(df_train, final_features)
X_val_clf, _ = prepare_X(df_val, final_features, cat_cols=cat_cols)

y_train_clf = (df_train[TARGET_NAME] > 0).astype(int)
y_val_clf = (df_val[TARGET_NAME] > 0).astype(int)

print("Train positives:", y_train_clf.sum())
print("Val positives:", y_val_clf.sum())
print("Categorical columns:", cat_cols)

clf_binary = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.02,

    num_leaves=31,
    max_depth=5,
    min_child_samples=50,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=1.0,
    reg_lambda=3.0,

    class_weight="balanced",

    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

clf_binary.fit(
    X_train_clf,
    y_train_clf,
    eval_set=[(X_val_clf, y_val_clf)],
    eval_metric="binary_logloss",
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
)

probs_val = clf_binary.predict_proba(X_val_clf)[:, 1]

roc_auc = roc_auc_score(y_val_clf, probs_val)
pr_auc = average_precision_score(y_val_clf, probs_val)

print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC :", round(pr_auc, 4))

threshold = 0.30
preds_val_clf = (probs_val >= threshold).astype(int)

print("\nClassification report, threshold =", threshold)
print(classification_report(y_val_clf, preds_val_clf))

cm = confusion_matrix(y_val_clf, preds_val_clf)
print("Confusion matrix:")
print(cm)
























df_train_reg = df_train[df_train[TARGET_NAME] > 0].copy()
df_val_reg = df_val[df_val[TARGET_NAME] > 0].copy()

print("Train regression rows:", df_train_reg.shape[0])
print("Val regression rows:", df_val_reg.shape[0])

# cap only by train
FX_UPPER_CAP = df_train_reg[TARGET_NAME].quantile(0.99)

print("FX upper cap:", FX_UPPER_CAP)

df_train_reg["FX_CAPPED"] = df_train_reg[TARGET_NAME].clip(upper=FX_UPPER_CAP)
df_val_reg["FX_CAPPED"] = df_val_reg[TARGET_NAME].clip(upper=FX_UPPER_CAP)

df_train_reg["FX_LOG"] = np.log1p(df_train_reg["FX_CAPPED"])
df_val_reg["FX_LOG"] = np.log1p(df_val_reg["FX_CAPPED"])

X_train_reg, _ = prepare_X(df_train_reg, final_features, cat_cols=cat_cols)
X_val_reg, _ = prepare_X(df_val_reg, final_features, cat_cols=cat_cols)

y_train_reg = df_train_reg["FX_LOG"]
y_val_reg = df_val_reg["FX_LOG"]

plt.figure(figsize=(8, 4))
sns.histplot(y_train_reg, bins=50)
plt.title("Regression target: log1p(FX capped)")
plt.xlabel("FX_LOG")
plt.ylabel("Count")
plt.show()

print(y_train_reg.describe())

reg = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=5000,
    learning_rate=0.02,

    num_leaves=31,
    max_depth=5,
    min_child_samples=30,

    subsample=0.8,
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
    callbacks=[
        lgb.early_stopping(200),
        lgb.log_evaluation(100)
    ]
)

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

print("=" * 60)
print("REGRESSION METRICS ON ACTIVE CLIENTS")
print("=" * 60)
print("MAE_log  :", round(mae_log, 5))
print("MedAE_log:", round(medae_log, 5))
print("R2_log   :", round(r2_log, 5))
print("-" * 60)
print("MAE original  :", round(mae_orig, 2))
print("MedAE original:", round(medae_orig, 2))
print("MAPE          :", round(mape, 4))
print("SMAPE         :", round(smape, 4))











#--------- FINAL
# P(FX > 0)
p_fx_val = clf_binary.predict_proba(X_val_clf)[:, 1]

# E(FX | FX > 0) для всіх validation клієнтів
X_val_all_reg, _ = prepare_X(df_val, final_features, cat_cols=cat_cols)

pred_log_cond_val = reg.predict(X_val_all_reg)
pred_log_cond_val = np.clip(pred_log_cond_val, 0, None)

pred_amount_cond_val = np.expm1(pred_log_cond_val)

# фінальний expected value
fx_expected_val = p_fx_val * pred_amount_cond_val

validation_results = df_val[[TARGET_NAME]].copy()
validation_results["P_FX"] = p_fx_val
validation_results["FX_COND_PRED"] = pred_amount_cond_val
validation_results["FX_EXPECTED"] = fx_expected_val

validation_results["FX_TRUE_ACTIVE"] = (validation_results[TARGET_NAME] > 0).astype(int)

validation_results = validation_results.rename(columns={
    TARGET_NAME: "FX_TRUE"
})

validation_results.head(10)





y_true_all = validation_results["FX_TRUE"].values
y_pred_all = validation_results["FX_EXPECTED"].values

mae_all = mean_absolute_error(y_true_all, y_pred_all)
medae_all = median_absolute_error(y_true_all, y_pred_all)

true_sum = y_true_all.sum()
pred_sum = y_pred_all.sum()
pred_true_ratio = pred_sum / max(true_sum, 1)

eps = 1e-9
smape_all = np.mean(
    2.0 * np.abs(y_true_all - y_pred_all) /
    np.maximum(np.abs(y_true_all) + np.abs(y_pred_all), eps)
)

print("=" * 70)
print("FINAL FX EXPECTED VALUE MODEL — VALIDATION")
print("=" * 70)
print("Rows:", len(validation_results))
print("True active clients:", int((y_true_all > 0).sum()))
print("Predicted total FX:", round(pred_sum, 2))
print("True total FX     :", round(true_sum, 2))
print("Pred / True ratio :", round(pred_true_ratio, 4))
print("-" * 70)
print("MAE all  :", round(mae_all, 2))
print("MedAE all:", round(medae_all, 2))
print("SMAPE all:", round(smape_all, 4))



plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=np.log1p(validation_results["FX_TRUE"]),
    y=np.log1p(validation_results["FX_EXPECTED"]),
    alpha=0.4
)
plt.xlabel("log1p(True FX)")
plt.ylabel("log1p(Predicted FX Expected)")
plt.title("True vs Predicted FX, log scale")
plt.grid(True, alpha=0.3)
plt.show()

























def get_lgb_importance(model, feature_names):
    importance_gain = model.booster_.feature_importance(importance_type="gain")

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_gain
    })

    total = df_imp["Importance"].sum()
    if total > 0:
        df_imp["Importance_%"] = 100 * df_imp["Importance"] / total
    else:
        df_imp["Importance_%"] = 0

    df_imp = df_imp.sort_values("Importance_%", ascending=False).reset_index(drop=True)

    return df_imp


clf_importance = get_lgb_importance(clf_binary, X_train_clf.columns)

print("Top-20 classifier features:")
print(clf_importance.head(20))

plt.figure(figsize=(10, 8))
sns.barplot(
    data=clf_importance.head(30),
    x="Importance_%",
    y="Feature"
)
plt.title("Classifier Feature Importance — FX activity")
plt.xlabel("Gain importance, %")
plt.ylabel("Feature")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



def get_lgb_importance(model, feature_names):
    importance_gain = model.booster_.feature_importance(importance_type="gain")

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_gain
    })

    total = df_imp["Importance"].sum()
    if total > 0:
        df_imp["Importance_%"] = 100 * df_imp["Importance"] / total
    else:
        df_imp["Importance_%"] = 0

    df_imp = df_imp.sort_values("Importance_%", ascending=False).reset_index(drop=True)

    return df_imp


clf_importance = get_lgb_importance(clf_binary, X_train_clf.columns)

print("Top-20 classifier features:")
print(clf_importance.head(20))

plt.figure(figsize=(10, 8))
sns.barplot(
    data=clf_importance.head(30),
    x="Importance_%",
    y="Feature"
)
plt.title("Classifier Feature Importance — FX activity")
plt.xlabel("Gain importance, %")
plt.ylabel("Feature")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()