import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. ПІДГОТОВКА ДАНИХ
# Повертаємось з логарифмів у гроші для бізнес-метрик
y_true_money = np.expm1(y_val_log)
y_pred_stretched_money = np.expm1(y_pred_stretched)
y_pred_original_money = np.expm1(y_pred_log)

print("=== CORRELATION METRICS ===")
# Pearson - лінійна (чутлива до викидів)
pearson_corr, _ = pearsonr(y_val_log, y_pred_stretched)
# Spearman - рангова (стійка, показує якість сортування)
spearman_corr, _ = spearmanr(y_val_log, y_pred_stretched)

print(f"Pearson Correlation (R): {pearson_corr:.4f}")
print(f"Spearman Correlation (Rank): {spearman_corr:.4f} (Має бути високою!)")
print("-" * 30)

# 2. LIFT ANALYSIS (Top 10% Performance)
def calculate_lift(y_true, y_pred, top_percent=0.1):
    df_res = pd.DataFrame({'true': y_true, 'pred': y_pred})
    # Сортуємо за прогнозом (беремо тих, кого модель вважає найбагатшими)
    df_res = df_res.sort_values('pred', ascending=False)
    
    n_top = int(len(df_res) * top_percent)
    top_segment = df_res.iloc[:n_top]
    
    avg_true_global = df_res['true'].mean()
    avg_true_top = top_segment['true'].mean()
    
    lift = avg_true_top / avg_true_global
    return lift, avg_true_top, avg_true_global

lift_score, top_mean, global_mean = calculate_lift(y_true_money, y_pred_stretched_money)

print("=== BUSINESS LIFT (Top 10%) ===")
print(f"Global Average Balance: {global_mean:,.0f} грн")
print(f"Model Top-10% Avg Balance: {top_mean:,.0f} грн")
print(f"LIFT SCORE: {lift_score:.2f}x")
print("(Це означає, що клієнти з топу моделі в Х разів багатші за середнього)")
print("-" * 30)

# 3. SEGMENTATION MATRIX (Low / Medium / High)
# Розбиваємо на 3 рівні частини (Terciles)
labels = ['Low', 'Medium', 'High']
df_bins = pd.DataFrame({
    'true': y_val_log,
    'pred': y_pred_stretched
})

# Створюємо класи на основі квантилів РЕАЛЬНИХ даних
df_bins['true_bin'] = pd.qcut(df_bins['true'], q=3, labels=labels)

# Створюємо класи на основі квантилів ПРОГНОЗУ (імітуємо скоринг)
df_bins['pred_bin'] = pd.qcut(df_bins['pred'], q=3, labels=labels)

cm = confusion_matrix(df_bins['true_bin'], df_bins['pred_bin'], labels=labels)

# Візуалізація
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=[f'Pred {l}' for l in labels],
            yticklabels=[f'True {l}' for l in labels])
plt.title("Segmentation Quality (Stretched Model)")
plt.ylabel("Actual Segment")
plt.xlabel("Predicted Segment")
plt.show()

# Точність по діагоналі
accuracy = np.trace(cm) / np.sum(cm)
print(f"Segmentation Accuracy: {accuracy:.2%}")