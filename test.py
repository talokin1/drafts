import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from scipy.stats import spearmanr

# ==========================================
# 1. ВАРІАНСНЕ РОЗТЯГУВАННЯ (THE LOGIC)
# ==========================================

# Рахуємо статистику
std_true = np.std(y_val_log)
std_pred = np.std(y_pred_log)
mean_true = np.mean(y_val_log)
mean_pred = np.mean(y_pred_log)

# Коефіцієнт розтягування
scaling_factor = std_true / std_pred

# Створюємо новий прогноз "Potential"
y_pred_stretched = (y_pred_log - mean_pred) * scaling_factor + mean_true

print(f"Scaling Factor applied: {scaling_factor:.4f}x")
print("-" * 60)

# ==========================================
# 2. ПОРІВНЯННЯ МЕТРИК (BEFORE vs AFTER)
# ==========================================

# Рахуємо метрики в Log-просторі
mae_orig = mean_absolute_error(y_val_log, y_pred_log)
medae_orig = median_absolute_error(y_val_log, y_pred_log)
r2_orig = r2_score(y_val_log, y_pred_log)

mae_new = mean_absolute_error(y_val_log, y_pred_stretched)
medae_new = median_absolute_error(y_val_log, y_pred_stretched)
r2_new = r2_score(y_val_log, y_pred_stretched)

# Рахуємо кореляцію рангів (якість сортування)
spearman_orig, _ = spearmanr(y_val_log, y_pred_log)
spearman_new, _ = spearmanr(y_val_log, y_pred_stretched)

print(f"{'METRIC':<15} | {'ORIGINAL':<12} | {'STRETCHED':<12} | {'CHANGE'}")
print("-" * 60)
print(f"{'MAE (log)':<15} | {mae_orig:<12.4f} | {mae_new:<12.4f} | {mae_new - mae_orig:+.4f} (Error ↑)")
print(f"{'MedAE (log)':<15} | {medae_orig:<12.4f} | {medae_new:<12.4f} | {medae_new - medae_orig:+.4f}")
print(f"{'R2 Score':<15} | {r2_orig:<12.4f} | {r2_new:<12.4f} | {r2_new - r2_orig:+.4f}")
print(f"{'Spearman Rank':<15} | {spearman_orig:<12.4f} | {spearman_new:<12.4f} | (Should be equal)")
print("-" * 60)

# ==========================================
# 3. ВІЗУАЛІЗАЦІЯ (DASHBOARD)
# ==========================================
plt.figure(figsize=(18, 6))

# ГРАФІК 1: Scatter Plot (Діагональ)
plt.subplot(1, 3, 1)
plt.scatter(y_val_log, y_pred_log, alpha=0.1, color='gray', label='Original (Conservative)')
plt.scatter(y_val_log, y_pred_stretched, alpha=0.2, color='blue', s=15, label='Stretched (Potential)')
# Лінія ідеального прогнозу
mn, mx = 0, 15 # Підлаштуй під свій діапазон
plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label='Ideal Line')
plt.title("Prediction vs Reality: Variance Inflation")
plt.xlabel("True Value (log)")
plt.ylabel("Predicted Value (log)")
plt.legend()
plt.grid(True, alpha=0.3)

# ГРАФІК 2: Distribution (KDE) - Відновлення форми
plt.subplot(1, 3, 2)
sns.kdeplot(y_val_log, color='red', label='True Distribution', fill=True, alpha=0.1)
sns.kdeplot(y_pred_log, color='gray', label='Original Pred (Collapsed)', linestyle="--")
sns.kdeplot(y_pred_stretched, color='blue', label='Stretched Pred (Restored)')
plt.title("Distribution Match (PDF)")
plt.xlabel("Log Amount")
plt.legend()
plt.grid(True, alpha=0.3)

# ГРАФІК 3: Segmentation Matrix (Бізнес-точність)
plt.subplot(1, 3, 3)
# Розбиваємо на 3 класи (Low/Mid/High)
labels = ['Low', 'Med', 'High']
df_bins = pd.DataFrame({'true': y_val_log, 'pred': y_pred_stretched})
df_bins['true_bin'] = pd.qcut(df_bins['true'], q=3, labels=labels)
df_bins['pred_bin'] = pd.qcut(df_bins['pred'], q=3, labels=labels)

cm = pd.crosstab(df_bins['true_bin'], df_bins['pred_bin'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title("Segmentation Accuracy (Stretched)")
plt.ylabel("Actual Segment")
plt.xlabel("Predicted Segment")

plt.tight_layout()
plt.show()