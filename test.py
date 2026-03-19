# --- Inference and Evaluation ---
p_vip_val = clf_router.predict_proba(X_val_final)[:, 1]

# Отримуємо прогнози регресорів у ЛОГ-просторі
pred_mass_log = reg_mass.predict(X_val_final)
pred_vip_log = reg_vip.predict(X_val_final)

# ВАЖЛИВО: Одразу переводимо ВСЕ в оригінальні гроші
pred_mass_original = np.expm1(pred_mass_log)
pred_vip_original = np.expm1(pred_vip_log)
y_val_original = np.expm1(y_val_log) # Повертаємо expm1 сюди!

# Зважуємо прогнози ВЖЕ В ОРИГІНАЛЬНИХ ГРОШАХ (Soft Blending)
pred_final_original = (p_vip_val * pred_vip_original) + ((1 - p_vip_val) * pred_mass_original)

# --- DataFrame ---
validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_val.index,
    'True_Value': y_val_original,
    'Predicted': pred_final_original
})

# --- Метрики ---
final_mae = mean_absolute_error(y_val_original, pred_final_original)
print(f"FINAL ORIGINAL MAE (Two-Stage Model): {final_mae:,.2f}")

from sklearn.metrics import roc_auc_score
print(f"Router ROC-AUC: {roc_auc_score(y_val_class, p_vip_val):.4f}")

# MAE тільки для Mass сегмента
mask_real_mass = y_val_original <= np.expm1(THRESHOLD_LOG)
mae_mass = mean_absolute_error(
    y_val_original[mask_real_mass], 
    pred_final_original[mask_real_mass]
)
print(f"MAE on REAL MASS clients : {mae_mass:,.2f}")

# MAE тільки для VIP сегмента
mask_real_vip = y_val_original > np.expm1(THRESHOLD_LOG)
mae_vip = mean_absolute_error(
    y_val_original[mask_real_vip], 
    pred_final_original[mask_real_vip]
)
print(f"MAE on REAL VIP clients  : {mae_vip:,.2f}")