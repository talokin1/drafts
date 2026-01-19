from sklearn.inspection import permutation_importance

# Беремо вашого навченого регресора і VIP-вибірку
# X_test_reg - це тільки ті об'єкти з тесту, де реально є гроші (>1000)
# y_test_reg_log - логарифм їхніх сум

mask_vip_test = y_test_cls == 1
X_test_reg = X_test[mask_vip_test]
y_test_reg_log = y_test_log[mask_vip_test]

print("Рахуємо Permutation Importance (це займе хвилину)...")
result = permutation_importance(
    regressor, X_test_reg, y_test_reg_log, 
    n_repeats=5, random_state=42, n_jobs=-1
)

# Сортуємо і виводимо
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(12, 8))
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False, labels=X_test_reg.columns[sorted_idx]
)
plt.title("Які фічі РЕАЛЬНО впливають на прогноз суми (Permutation Importance)")
plt.tight_layout()
plt.show()