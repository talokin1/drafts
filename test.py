# Рахуємо середній залишок тільки для VIP-клієнтів у тесті
vip_mean = np.expm1(y_test_log[y_test_cls == 1]).mean()
vip_median = np.expm1(y_test_log[y_test_cls == 1]).median()

# "Наївний прогноз": всім прогнозуємо просто медіану
naive_preds = np.full(len(final_preds), vip_median)
naive_mae = mean_absolute_error(np.expm1(y_test_log), naive_preds)

print(f"Середній чек VIP-клієнта: {vip_mean:.2f} грн")
print(f"Медіана VIP-клієнта: {vip_median:.2f} грн")
print(f"Помилка вашої моделі: {final_mae:.2f} грн")
print(f"Помилка наївної моделі (просто медіана): {naive_mae:.2f} грн")