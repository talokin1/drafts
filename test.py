import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, mean_absolute_percentage_error

# =====================================================================
# 1. ТРЕНУВАННЯ МОДЕЛЕЙ (З МЕТОДОМ 2: Зважування клієнтів)
# =====================================================================

print("--- Training Router Classifier (Cost-Sensitive) ---")
clf_router = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=30,
    random_state=RANDOM_STATE, 
    n_jobs=-1
)

# Перетворюємо логарифми таргета у ваги (чим більший прибуток, тим більша вага)
train_weights = y_train_log.to_numpy()
val_weights = y_val_log.to_numpy()

clf_router.fit(
    X_train_final, y_train_class,
    sample_weight=train_weights,             # МЕТОД 2: Ваги для тренування
    eval_set=[(X_val_final, y_val_class)],
    eval_sample_weight=[val_weights],        # МЕТОД 2: Ваги для Early Stopping
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
reg_vip.fit(
    X_train_vip, np.expm1(y_train_vip), 
    eval_set=[(X_val_vip, np.expm1(y_val_vip_log))],
    eval_metric="l1", 
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)

# =====================================================================
# 2. ІНФЕРЕНС ТА ОПТИМІЗАЦІЯ ПОРОГУ (МЕТОД 1: Cost-Sensitive Thresholding)
# =====================================================================
print("\n--- Inference and Threshold Optimization ---")

p_vip_val = clf_router.predict_proba(X_val_final)[:, 1]

# Отримуємо ізольовані прогнози в оригінальних грошах
pred_mass_orig = np.expm1(reg_mass.predict(X_val_final)) 
pred_vip_orig = reg_vip.predict(X_val_final) 

y_val_original = np.expm1(y_val_log)

# Шукаємо математичний оптимум для порогу
print("Шукаємо оптимальний поріг...")
thresholds = np.linspace(0.1, 0.9, 100) # Перевіряємо 100 варіантів від 0.1 до 0.9
maes = []

for t in thresholds:
    # Симулюємо жорсткий вибір для кожного порогу
    temp_pred = np.where(p_vip_val > t, pred_vip_orig, pred_mass_orig)
    maes.append(mean_absolute_error(y_val_original, temp_pred))

# Знаходимо індекс найменшої помилки
best_idx = np.argmin(maes)
best_threshold = thresholds[best_idx]
best_mae = maes[best_idx]

print(f"Математично оптимальний поріг Роутера: {best_threshold:.4f}")

# Застосовуємо найкращий поріг для фінальних прогнозів
pred_final_original = np.where(p_vip_val > best_threshold, pred_vip_orig, pred_mass_orig)

# =====================================================================
# 3. ОЦІНКА РЕЗУЛЬТАТІВ
# =====================================================================
print("\n--- Загальні метрики пайплайну ---")
router_auc = roc_auc_score(y_val_class, p_vip_val)
final_mae = mean_absolute_error(y_val_original, pred_final_original)

print(f"Router ROC-AUC: {router_auc:.4f}")
print(f"FINAL ORIGINAL MAE (Optimized Hard Routing): {final_mae:,.2f}")

mask_real_mass = y_val_original <= np.expm1(THRESHOLD_LOG)
mask_real_vip = y_val_original > np.expm1(THRESHOLD_LOG)

mae_pipeline_mass = mean_absolute_error(y_val_original[mask_real_mass], pred_final_original[mask_real_mass])
mae_pipeline_vip = mean_absolute_error(y_val_original[mask_real_vip], pred_final_original[mask_real_vip])

print(f"\n--- Метрики пайплайну по фактичних сегментах ---")
print(f"MAE пайплайну на REAL MASS: {mae_pipeline_mass:,.2f}")
print(f"MAE пайплайну на REAL VIP : {mae_pipeline_vip:,.2f}")

validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_val_final.index,
    'True_Value': y_val_original,
    'Predicted': pred_final_original,
    'Router_Prob_VIP': p_vip_val,
    'Pred_Mass_Only': pred_mass_orig,
    'Pred_VIP_Only': pred_vip_orig
})