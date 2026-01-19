import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Виберемо топ-3 фічі, які (логічно) мають впливати на наявність грошей
# Наприклад: Оборот, Попередній кеш, Активи (заміни на свої реальні назви колонок!)
top_features = ['REVENUE_CUR', 'CASH_PREV', 'ASSETS_DIF'] 

thresholds = [0, 1, 5, 10, 20, 50, 100, 200]
correlations = {feat: [] for feat in top_features}

for t in thresholds:
    # Створюємо бінарний таргет для цього порогу
    binary_target = (df['CURR_ACC'] > t).astype(int)
    
    for feat in top_features:
        # Рахуємо кореляцію Спірмена (бо розподіли не нормальні)
        corr = df[feat].corr(binary_target, method='spearman')
        correlations[feat].append(corr)

# Візуалізація
plt.figure(figsize=(10, 6))
for feat in top_features:
    plt.plot(thresholds, correlations[feat], marker='o', label=feat)

plt.title('Як змінюється "читабельність" клієнта залежно від порогу')
plt.xlabel('Поріг відсікання (грн)')
plt.ylabel('Кореляція фічі з класом (Spearman)')
plt.legend()
plt.grid(True)
plt.show()




plt.figure(figsize=(12, 5))

# Беремо тільки тих, у кого є хоч копійка, і логарифмуємо
# log10 зручніше для сприйняття (1 = 10 грн, 2 = 100 грн, -1 = 0.1 грн)
log_data = np.log10(df[df['CURR_ACC'] > 0]['CURR_ACC'])

sns.histplot(log_data, bins=100, kde=True)

# Додаємо мітки для зрозумілості
plt.xticks([-1, 0, 1, 2, 3], ['0.1 грн', '1 грн', '10 грн', '100 грн', '1000 грн'])
plt.title('Розподіл залишків у логарифмічній шкалі')
plt.xlabel('Сума (Log10)')
plt.axvline(x=np.log10(10), color='r', linestyle='--', label='Поріг 10 грн')
plt.legend()
plt.show()




stats = []
total_rows = len(df)

for t in [0, 5, 10, 50, 100]:
    count_active = (df['CURR_ACC'] > t).sum()
    share = count_active / total_rows * 100
    stats.append({'Threshold': t, 'Active_Clients': count_active, 'Share_%': share})

pd.DataFrame(stats)