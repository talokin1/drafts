# Перевірка ТІЛЬКИ на реальних VIP-клієнтах з тестового набору
mask_vip_test = y_test_cls == 1
X_test_vip = X_test[mask_vip_test]
y_test_vip_log = y_test_log[mask_vip_test]

# Прогноз чистого регресора
pred_log_vip = regressor.predict(X_test_vip)
pred_amount_vip = np.expm1(pred_log_vip)
true_amount_vip = np.expm1(y_test_vip_log)

r2_vip = r2_score(true_amount_vip, pred_amount_vip)
print(f"R2 регресора в ідеальних умовах: {r2_vip:.4f}")