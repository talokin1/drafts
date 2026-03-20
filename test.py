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