import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score

# =====================================================================
# 1. ТРЕНУВАННЯ МОДЕЛЕЙ (БАЗОВА ВЕРСІЯ)
# =====================================================================

print("--- Training Router Classifier ---")
clf_router = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=30,
    random_state=RANDOM_STATE, 
    n_jobs=-1
)

clf_router.fit(
    X_train_final, y_train_class,
    eval_set=[(X_val_final, y_val_class)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

print("\n--- Training Mass Regressor (TARGET: LOG) ---")
reg_mass = lgb.LGBMRegressor(
    objective="regression_l1", 
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=20,
    random_state=RANDOM_STATE
)

reg_mass.fit(
    X_train_mass, y_train_mass,
    eval_set=[(X_val_mass, y_val_mass_log)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)

print("\n--- Training VIP Regressor (TARGET: ORIGINAL MONEY) ---")
reg_vip = lgb.LGBMRegressor(
    objective="tweedie", 
    tweedie_variance_power=1.5,
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=15,
    min_child_samples=10,
    random_state=RANDOM_STATE
)

# Експоненціюємо таргет прямо під час передачі у fit
reg_vip.fit(
    X_train_vip, np.expm1(y_train_vip), 
    eval_set=[(X_val_vip, np.expm1(y_val_vip_log))],
    eval_metric="l1", 
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)


# =====================================================================
# 2. ІНФЕРЕНС ТА ЖОРСТКИЙ ВИБІР (HARD ROUTING)
# =====================================================================
print("\n--- Inference and Hard Routing ---")

# Отримуємо ймовірність, що клієнт належить до VIP-сегменту
p_vip_val = clf_router.predict_proba(X_val_final)[:, 1]

# Отримуємо ізольовані прогнози
pred_mass_orig = np.expm1(reg_mass.predict(X_val_final)) # Експоненціюємо логарифми
pred_vip_orig = reg_vip.predict(X_val_final)             # Вже в грошах (Tweedie)

# ЖОРСТКИЙ ВИБІР: Якщо ймовірність VIP вища за поріг, беремо прогноз VIP
# Оскільки хибне спрацьовування Роутера "вбиває" MASS-прогноз (як ми побачили), 
# ми маємо зробити поріг досить суворим. Почнемо з 0.5, але його варто покрутити.
ROUTER_THRESHOLD = 0.5
pred_final_original = np.where(p_vip_val > ROUTER_THRESHOLD, pred_vip_orig, pred_mass_orig)

# Відновлюємо справжній таргет у грошах
y_val_original = np.expm1(y_val_log)


# =====================================================================
# 3. ОЦІНКА РЕЗУЛЬТАТІВ (EVALUATION)
# =====================================================================
print("\n--- Загальні метрики пайплайну ---")
router_auc = roc_auc_score(y_val_class, p_vip_val)
final_mae = mean_absolute_error(y_val_original, pred_final_original)

print(f"Router ROC-AUC: {router_auc:.4f}")
print(f"FINAL ORIGINAL MAE (Hard Routing): {final_mae:,.2f}")

# Оцінка якості на фактичних сегментах
mask_real_mass = y_val_original <= np.expm1(THRESHOLD_LOG)
mask_real_vip = y_val_original > np.expm1(THRESHOLD_LOG)

mae_pipeline_mass = mean_absolute_error(y_val_original[mask_real_mass], pred_final_original[mask_real_mass])
mae_pipeline_vip = mean_absolute_error(y_val_original[mask_real_vip], pred_final_original[mask_real_vip])

print(f"\n--- Метрики пайплайну по фактичних сегментах ---")
print(f"MAE пайплайну на REAL MASS: {mae_pipeline_mass:,.2f}")
print(f"MAE пайплайну на REAL VIP : {mae_pipeline_vip:,.2f}")

# Датафрейм для дебагу
validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_val_final.index,
    'True_Value': y_val_original,
    'Predicted': pred_final_original,
    'Router_Prob_VIP': p_vip_val,
    'Pred_Mass_Only': pred_mass_orig,
    'Pred_VIP_Only': pred_vip_orig
})