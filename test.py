import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score
)

# =========================================================
# 0. INPUT DATA
# =========================================================
# Очікується:
# X_ -> pd.DataFrame з фічами
# y_original -> pd.Series або np.array з таргетом у ГРОШАХ
#
# Приклад:
# X_ = df_features.copy()
# y_original = df["TARGET"].copy()

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD_LOG = 6.0   # стартова межа між MASS і VIP
N_QCUT_BINS = 10

# ---------------------------------------------------------
# Захист від неправильних типів
# ---------------------------------------------------------
if not isinstance(X_, pd.DataFrame):
    X_ = pd.DataFrame(X_)

if isinstance(y_original, np.ndarray):
    y_original = pd.Series(y_original, index=X_.index, name="target")
elif not isinstance(y_original, pd.Series):
    y_original = pd.Series(y_original, index=X_.index, name="target")

# На випадок пропусків / негативів
if (y_original < 0).any():
    raise ValueError("y_original містить від'ємні значення. log1p для такого таргету некоректний.")

# =========================================================
# 1. LOG TARGET
# =========================================================
y_log = np.log1p(y_original)

# =========================================================
# 2. TRAIN / VAL SPLIT
# =========================================================
# Стратифікація по квантилях лог-таргету
y_bins = pd.qcut(y_log, q=N_QCUT_BINS, labels=False, duplicates="drop")

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X_,
    y_log,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_bins
)

# Індекс збережемо для аналізу
train_index = X_train.index
val_index = X_val.index

# =========================================================
# 3. CATEGORICAL FEATURES
# =========================================================
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ["object", "category"]]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

    # Узгоджуємо категорії train -> val
    X_val[c] = X_val[c].cat.set_categories(X_train[c].cat.categories)

print("Categorical columns:", len(cat_cols))
print(cat_cols[:20], "..." if len(cat_cols) > 20 else "")

# =========================================================
# 4. CLASS TARGET FOR ROUTER
# =========================================================
y_train_class = (y_train_log > THRESHOLD_LOG).astype(int)
y_val_class = (y_val_log > THRESHOLD_LOG).astype(int)

print("\n--- Segment sizes ---")
print("Train MASS:", int((y_train_class == 0).sum()))
print("Train VIP :", int((y_train_class == 1).sum()))
print("Val MASS  :", int((y_val_class == 0).sum()))
print("Val VIP   :", int((y_val_class == 1).sum()))

# =========================================================
# 5. MASKS FOR SEGMENT REGRESSORS
# =========================================================
mask_train_mass = (y_train_log <= THRESHOLD_LOG)
mask_train_vip = (y_train_log > THRESHOLD_LOG)

mask_val_mass = (y_val_log <= THRESHOLD_LOG)
mask_val_vip = (y_val_log > THRESHOLD_LOG)

# =========================================================
# 6. ROUTER CLASSIFIER
# =========================================================
print("\n--- Training Router Classifier ---")

# баланс класів, якщо VIP мало
pos = int((y_train_class == 1).sum())
neg = int((y_train_class == 0).sum())
scale_pos_weight = neg / pos if pos > 0 else 1.0

clf_router = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

clf_router.fit(
    X_train,
    y_train_class,
    eval_set=[(X_val, y_val_class)],
    eval_metric="binary_logloss",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)

p_vip_val = clf_router.predict_proba(X_val)[:, 1]
router_auc = roc_auc_score(y_val_class, p_vip_val)

print(f"Router ROC-AUC: {router_auc:.4f}")

# =========================================================
# 7. MASS REGRESSOR (LOG SPACE)
# =========================================================
print("\n--- Training MASS Regressor (LOG SPACE) ---")

reg_mass = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg_mass.fit(
    X_train.loc[mask_train_mass],
    y_train_log.loc[mask_train_mass],
    eval_set=[(X_val.loc[mask_val_mass], y_val_log.loc[mask_val_mass])] if mask_val_mass.sum() > 0 else None,
    eval_metric="l1",
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)

# =========================================================
# 8. VIP REGRESSOR (LOG SPACE)
# =========================================================
print("\n--- Training VIP Regressor (LOG SPACE) ---")

reg_vip = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.015,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg_vip.fit(
    X_train.loc[mask_train_vip],
    y_train_log.loc[mask_train_vip],
    eval_set=[(X_val.loc[mask_val_vip], y_val_log.loc[mask_val_vip])] if mask_val_vip.sum() > 0 else None,
    eval_metric="l1",
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)

# =========================================================
# 9. PREDICTIONS IN LOG SPACE
# =========================================================
pred_mass_log = reg_mass.predict(X_val)
pred_vip_log = reg_vip.predict(X_val)

# захист від дивних негативних лог-прогнозів
pred_mass_log = np.clip(pred_mass_log, a_min=0, a_max=None)
pred_vip_log = np.clip(pred_vip_log, a_min=0, a_max=None)

# =========================================================
# 10. SOFT ROUTING
# =========================================================
pred_final_soft_log = p_vip_val * pred_vip_log + (1.0 - p_vip_val) * pred_mass_log

# =========================================================
# 11. HARD ROUTING + BEST THRESHOLD SEARCH
# =========================================================
threshold_grid = np.arange(0.05, 0.96, 0.05)

y_val_original = np.expm1(y_val_log.values)

best_hard_threshold = None
best_hard_mae = np.inf
best_hard_pred_original = None

for t in threshold_grid:
    pred_hard_log = np.where(p_vip_val > t, pred_vip_log, pred_mass_log)
    pred_hard_original = np.expm1(pred_hard_log)

    mae_t = mean_absolute_error(y_val_original, pred_hard_original)

    if mae_t < best_hard_mae:
        best_hard_mae = mae_t
        best_hard_threshold = t
        best_hard_pred_original = pred_hard_original.copy()

# =========================================================
# 12. BACK TO ORIGINAL MONEY
# =========================================================
pred_mass_original = np.expm1(pred_mass_log)
pred_vip_original = np.expm1(pred_vip_log)
pred_final_soft_original = np.expm1(pred_final_soft_log)

# =========================================================
# 13. METRICS
# =========================================================
def print_regression_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"MAE      : {mae:,.2f}")
    print(f"MedAE    : {med_ae:,.2f}")
    print(f"R2       : {r2:.4f}")

print("\n================ FINAL METRICS ================")
print(f"Router ROC-AUC        : {router_auc:.4f}")
print(f"Best hard threshold   : {best_hard_threshold:.2f}")
print(f"Best hard routing MAE : {best_hard_mae:,.2f}")

print_regression_metrics("SOFT ROUTING", y_val_original, pred_final_soft_original)
print_regression_metrics("MASS ONLY", y_val_original, pred_mass_original)
print_regression_metrics("VIP ONLY", y_val_original, pred_vip_original)
print_regression_metrics("BEST HARD ROUTING", y_val_original, best_hard_pred_original)

# =========================================================
# 14. SEGMENT METRICS BY TRUE SEGMENT
# =========================================================
real_mass_mask = y_val_original <= np.expm1(THRESHOLD_LOG)
real_vip_mask = y_val_original > np.expm1(THRESHOLD_LOG)

def safe_segment_mae(y_true, y_pred, mask, name):
    if mask.sum() == 0:
        print(f"{name}: no rows")
        return
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    print(f"{name}: {mae:,.2f}")

print("\n================ SEGMENT MAE ================")
print("SOFT ROUTING:")
safe_segment_mae(y_val_original, pred_final_soft_original, real_mass_mask, "  Real MASS MAE")
safe_segment_mae(y_val_original, pred_final_soft_original, real_vip_mask, "  Real VIP  MAE")

print("\nBEST HARD ROUTING:")
safe_segment_mae(y_val_original, best_hard_pred_original, real_mass_mask, "  Real MASS MAE")
safe_segment_mae(y_val_original, best_hard_pred_original, real_vip_mask, "  Real VIP  MAE")

# =========================================================
# 15. VALIDATION RESULTS TABLE
# =========================================================
validation_results = pd.DataFrame({
    "IDENTIFYCODE": val_index,
    "True_Value": y_val_original,
    "Pred_Soft": pred_final_soft_original,
    "Pred_Hard_Best": best_hard_pred_original,
    "Router_Prob_VIP": p_vip_val,
    "Pred_Mass_Only": pred_mass_original,
    "Pred_VIP_Only": pred_vip_original,
    "True_Log": y_val_log.values,
    "Pred_Mass_Log": pred_mass_log,
    "Pred_VIP_Log": pred_vip_log,
    "Pred_Soft_Log": pred_final_soft_log,
    "True_Segment": np.where(real_vip_mask, "VIP", "MASS"),
    "Pred_Segment_Hard_Best": np.where(p_vip_val > best_hard_threshold, "VIP", "MASS")
})

validation_results["Abs_Error_Soft"] = np.abs(validation_results["True_Value"] - validation_results["Pred_Soft"])
validation_results["Abs_Error_Hard_Best"] = np.abs(validation_results["True_Value"] - validation_results["Pred_Hard_Best"])
validation_results["Signed_Error_Soft"] = validation_results["Pred_Soft"] - validation_results["True_Value"]
validation_results["Signed_Error_Hard_Best"] = validation_results["Pred_Hard_Best"] - validation_results["True_Value"]

validation_results["Bias_Soft"] = np.where(
    validation_results["Signed_Error_Soft"] > 0, "Over",
    np.where(validation_results["Signed_Error_Soft"] < 0, "Under", "Exact")
)

validation_results["Bias_Hard_Best"] = np.where(
    validation_results["Signed_Error_Hard_Best"] > 0, "Over",
    np.where(validation_results["Signed_Error_Hard_Best"] < 0, "Under", "Exact")
)

print("\nValidation results shape:", validation_results.shape)
print(validation_results.head())

# =========================================================
# 16. TOP ERRORS
# =========================================================
top_soft_errors = validation_results.sort_values("Abs_Error_Soft", ascending=False).head(50)
top_hard_errors = validation_results.sort_values("Abs_Error_Hard_Best", ascending=False).head(50)

print("\n--- TOP 20 SOFT ROUTING ERRORS ---")
print(
    top_soft_errors[
        [
            "IDENTIFYCODE", "True_Value", "Pred_Soft", "Router_Prob_VIP",
            "Pred_Mass_Only", "Pred_VIP_Only", "True_Segment",
            "Abs_Error_Soft", "Signed_Error_Soft", "Bias_Soft"
        ]
    ].head(20)
)

print("\n--- TOP 20 HARD ROUTING ERRORS ---")
print(
    top_hard_errors[
        [
            "IDENTIFYCODE", "True_Value", "Pred_Hard_Best", "Router_Prob_VIP",
            "Pred_Mass_Only", "Pred_VIP_Only", "True_Segment",
            "Pred_Segment_Hard_Best",
            "Abs_Error_Hard_Best", "Signed_Error_Hard_Best", "Bias_Hard_Best"
        ]
    ].head(20)
)

# =========================================================
# 17. QUICK BIAS SUMMARY
# =========================================================
print("\n================ BIAS SUMMARY ================")
print("SOFT ROUTING:")
print(validation_results["Bias_Soft"].value_counts(dropna=False))

print("\nBEST HARD ROUTING:")
print(validation_results["Bias_Hard_Best"].value_counts(dropna=False))

# =========================================================
# 18. OPTIONAL: FEATURE IMPORTANCE
# =========================================================
router_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance_gain": clf_router.booster_.feature_importance(importance_type="gain")
}).sort_values("importance_gain", ascending=False)

mass_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance_gain": reg_mass.booster_.feature_importance(importance_type="gain")
}).sort_values("importance_gain", ascending=False)

vip_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance_gain": reg_vip.booster_.feature_importance(importance_type="gain")
}).sort_values("importance_gain", ascending=False)

print("\n--- Router feature importance ---")
print(router_importance.head(20))

print("\n--- MASS regressor feature importance ---")
print(mass_importance.head(20))

print("\n--- VIP regressor feature importance ---")
print(vip_importance.head(20))

# =========================================================
# 19. OPTIONAL SAVE RESULTS
# =========================================================
# validation_results.to_csv("validation_results_moe.csv", index=False)
# router_importance.to_csv("router_importance.csv", index=False)
# mass_importance.to_csv("mass_importance.csv", index=False)
# vip_importance.to_csv("vip_importance.csv", index=False)