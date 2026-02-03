import numpy as np
import matplotlib.pyplot as plt

# 1. Беремо прогнози твоєї найкращої моделі (MAE/Regression)
# y_val_log - це реальні дані (тест)
# y_pred_log - це те, що видала модель (стиснута хмара)

# 2. Рахуємо статистику
std_true = np.std(y_val_log)
std_pred = np.std(y_pred_log)
mean_true = np.mean(y_val_log)
mean_pred = np.mean(y_pred_log)

print(f"Std True: {std_true:.4f}")
print(f"Std Pred: {std_pred:.4f}")
print(f"Ми маємо розтягнути прогноз в {std_true/std_pred:.2f} разів!")

# 3. РОЗТЯГУВАННЯ (SCALING)
# Формула: (X - mean) * (std_target / std_current) + mean_target
scaling_factor = std_true / std_pred
y_pred_stretched = (y_pred_log - mean_pred) * scaling_factor + mean_true

# 4. Візуалізація результату
plt.figure(figsize=(10, 8))

# Малюємо оригінальну "нудну" хмару (сірим)
plt.scatter(y_val_log, y_pred_log, alpha=0.1, color='gray', label='Original Prediction')

# Малюємо НОВУ розтягнуту хмару (синім)
plt.scatter(y_val_log, y_pred_stretched, alpha=0.3, color='blue', s=15, label='Stretched (Potential)')

# Ідеальна лінія
mn, mx = 0, 14
plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label='Ideal Diagonal')

plt.title("Variance Inflation: Matching the Potential")
plt.xlabel("True CURR_ACC (log)")
plt.ylabel("Predicted Potential (log)")
plt.legend()
plt.show()

# 5. Метрики (подивимось, що сталось)
from sklearn.metrics import r2_score, mean_absolute_error
print("Original R2:", r2_score(y_val_log, y_pred_log))
print("Stretched R2:", r2_score(y_val_log, y_pred_stretched))