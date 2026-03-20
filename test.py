import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, mean_absolute_percentage_error

# =====================================================================
# 1. ТРЕНУВАННЯ МОДЕЛЕЙ
# =====================================================================

print("--- Training Router Classifier ---")
clf_router = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=30,
    random_state=RANDOM_STATE, # Переконайся, що ця змінна визначена вище
    n_jobs=-1
)

clf_router.fit(
    X_train_final, y_train_class,
    eval_set=[(X_val_final, y_val_class)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

print("\n--- Training Mass Regressor (TARGET: LOG) ---")
# Mass-сегмент має меншу дисперсію, логарифм + L1 (MAE) тут працює чудово
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
# VIP-сегмент має важкий правий хвіст. 
# Навчаємо на грошах за допомогою розподілу Tweedie, щоб уникнути зміщення медіани
reg_vip = lgb.LGBMRegressor(
    objective="tweedie", 
    tweedie_variance_power=1.5, # Оптимально для фінансових величин (між Poisson та Gamma)
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=15,
    min_child_samples=10,
    random_state=RANDOM_STATE
)

# УВАГА: np.expm1 застосовується до таргету прямо під час навчання
reg_vip.fit(
    X_train_vip, np.expm1(y_train_vip), 
    eval_set=[(X_val_vip, np.expm1(y_val_vip_log))],
    eval_metric="l1", # Використовуємо MAE для Early Stopping
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)


# =====================================================================
# 2. ІНФЕРЕНС ТА ЗВАЖУВАННЯ (SOFT BLENDING)
# =====================================================================
print("\n--- Inference and Soft Blending ---")

# Отримуємо ймовірність, що клієнт належить до VIP-сегменту
p_vip_val = clf_router.predict_proba(X_val_final)[:, 1]

# Отримуємо прогнози регресорів
# Для Mass робимо експоненціювання, бо модель вчилася на логарифмах
pred_mass_orig = np.expm1(reg_mass.predict(X_val_final)) 

# Для VIP експоненціювання НЕ РОБИМО, бо модель вчилася на оригінальних грошах
pred_vip_orig = reg_vip.predict(X_val_final) 

# Об'єднуємо прогнози через зважене математичне сподівання
pred_final_original = (p_vip_val * pred_vip_orig) + ((1 - p_vip_val) * pred_mass_orig)

# Відновлюємо справжній таргет у грошах для оцінки
y_val_original = np.expm1(y_val_log)


# =====================================================================
# 3. ОЦІНКА РЕЗУЛЬТАТІВ (EVALUATION)
# =====================================================================
print("\n--- Загальні метрики пайплайну ---")
router_auc = roc_auc_score(y_val_class, p_vip_val)
final_mae = mean_absolute_error(y_val_original, pred_final_original)

print(f"Router ROC-AUC: {router_auc:.4f}")
print(f"FINAL ORIGINAL MAE (Soft Blending): {final_mae:,.2f}")

# Оцінка якості на фактичних сегментах (щоб розуміти, де модель помиляється найбільше)
mask_real_mass = y_val_original <= np.expm1(THRESHOLD_LOG)
mask_real_vip = y_val_original > np.expm1(THRESHOLD_LOG)

mae_pipeline_mass = mean_absolute_error(y_val_original[mask_real_mass], pred_final_original[mask_real_mass])
mae_pipeline_vip = mean_absolute_error(y_val_original[mask_real_vip], pred_final_original[mask_real_vip])

print(f"\n--- Метрики пайплайну по фактичних сегментах ---")
print(f"MAE пайплайну на REAL MASS: {mae_pipeline_mass:,.2f}")
print(f"MAE пайплайну на REAL VIP : {mae_pipeline_vip:,.2f}")

# Збираємо датафрейм для побудови відер (Buckets) та дебагу
validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_val_final.index,
    'True_Value': y_val_original,
    'Predicted': pred_final_original,
    'Router_Prob_VIP': p_vip_val,
    'Pred_Mass_Only': pred_mass_orig,
    'Pred_VIP_Only': pred_vip_orig
})