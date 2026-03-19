import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score, mean_absolute_percentage_error

print("\n--- Inference and Evaluation ---")

# 1. Отримуємо прогнози від усіх трьох моделей у логарифмічному просторі
p_vip_val = clf_router.predict_proba(X_val_final)[:, 1]
pred_mass_log = reg_mass.predict(X_val_final)
pred_vip_log = reg_vip.predict(X_val_final)

# 2. Hard Routing (Жорсткий вибір)
# Якщо Роутер впевнений, що це VIP (>0.5), беремо прогноз VIP-моделі, інакше - Mass
ROUTER_THRESHOLD = 0.5
pred_final_log_hard = np.where(p_vip_val > ROUTER_THRESHOLD, pred_vip_log, pred_mass_log)

# 3. Переводимо прогнози та справжні значення в оригінальний простір (гроші)
pred_final_original = np.expm1(pred_final_log_hard)
y_val_original = np.expm1(y_val_log)

# Збираємо результати в DataFrame для зручного дебагу
validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_val_final.index,
    'True_Value': y_val_original,
    'Predicted': pred_final_original,
    'Router_Prob_VIP': p_vip_val
})

# ==========================================
# ОЦІНКА РЕЗУЛЬТАТІВ
# ==========================================

# 4. Загальні метрики всього пайплайну
final_mae = mean_absolute_error(y_val_original, pred_final_original)
router_auc = roc_auc_score(y_val_class, p_vip_val)

print(f"--- Загальні метрики пайплайну ---")
print(f"Router ROC-AUC: {router_auc:.4f}")
print(f"FINAL ORIGINAL MAE (Two-Stage Pipeline): {final_mae:,.2f}")

# 5. Детальний аналіз: як працює пайплайн на різних типах клієнтів
mask_real_mass = y_val_original <= np.expm1(THRESHOLD_LOG)
mask_real_vip = y_val_original > np.expm1(THRESHOLD_LOG)

mae_pipeline_mass = mean_absolute_error(y_val_original[mask_real_mass], pred_final_original[mask_real_mass])
mae_pipeline_vip = mean_absolute_error(y_val_original[mask_real_vip], pred_final_original[mask_real_vip])

print(f"\n--- Метрики пайплайну по фактичних сегментах ---")
print(f"MAE пайплайну на REAL MASS: {mae_pipeline_mass:,.2f}")
print(f"MAE пайплайну на REAL VIP : {mae_pipeline_vip:,.2f}")

# 6. ІЗОЛЬОВАНА ПЕРЕВІРКА (Ідеальний Роутер)
# Оцінюємо регресори так, ніби Роутер ніколи не помиляється.
# Це покаже "стелю" якості твоїх регресійних моделей.
pred_ideal_mass_orig = np.expm1(reg_mass.predict(X_val_mass))
pred_ideal_vip_orig = np.expm1(reg_vip.predict(X_val_vip))

mae_ideal_mass = mean_absolute_error(np.expm1(y_val_mass_log), pred_ideal_mass_orig)
mae_ideal_vip = mean_absolute_error(np.expm1(y_val_vip_log), pred_ideal_vip_orig)

print(f"\n--- Ізольовані метрики регресорів (за умови ідеального роутера) ---")
print(f"Ізольований MAE чистої MASS моделі: {mae_ideal_mass:,.2f}")
print(f"Ізольований MAE чистої VIP моделі : {mae_ideal_vip:,.2f}")

# Додаємо MAPE для VIP сегменту, щоб інтуїтивно розуміти помилку у відсотках
mape_ideal_vip = mean_absolute_percentage_error(np.expm1(y_val_vip_log), pred_ideal_vip_orig)
print(f"Ізольований MAPE чистої VIP моделі: {mape_ideal_vip:.2%}")