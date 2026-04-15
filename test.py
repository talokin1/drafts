import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Налаштування стилю для графіків (виглядає професійно для звітів)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11})

# ==========================================
# 1. ПІДГОТОВКА ДАНИХ
# ==========================================
# Формуємо таргет (за твоєю логікою)
dataset["BALANCE"] = (dataset["AMOUNT_SAVINGS"].fillna(0) + 
                      dataset["BALANCE_CURRENT_ACCOUNTS"].fillna(0) + 
                      dataset["BALANCE_DEBIT_CARDS"].fillna(0) + 
                      dataset["BALANCE_SALARY_CARDS"].fillna(0) + 
                      dataset["BALANCE_SOCIAL_CARDS"].fillna(0))

current_hard_threshold = 800000

# ==========================================
# 2. МАТЕМАТИЧНІ РОЗРАХУНКИ
# ==========================================

# А. Аналіз існуючого сегмента HNWI (Overlap Analysis)
hnwi_mask = dataset['SEGMENT'] == 'HNWI'
hnwi_balances = dataset[hnwi_mask]['BALANCE']

p10_hnwi = hnwi_balances.quantile(0.10)
p25_hnwi = hnwi_balances.quantile(0.25)
median_hnwi = hnwi_balances.median()

# Б. Аналіз концентрації капіталу (Крива Лоренца / Парето)
df_sorted = dataset[['BALANCE']].copy().sort_values('BALANCE').reset_index(drop=True)
df_sorted['cum_clients'] = (df_sorted.index + 1) / len(df_sorted)
df_sorted['cum_balance'] = df_sorted['BALANCE'].cumsum() / df_sorted['BALANCE'].sum()

# Шукаємо пороги для топ-1%, 3%, 5% найбагатших
targets = {
    "Топ-1%": 0.99, 
    "Топ-3%": 0.97, 
    "Топ-5%": 0.95
}
concentration_metrics = {}

for label, t in targets.items():
    # Знаходимо першого клієнта, який перетинає цей перцентиль
    idx = df_sorted[df_sorted['cum_clients'] >= t].index[0]
    val = df_sorted.loc[idx, 'BALANCE']
    # Рахуємо, скільки грошей у тих, хто вище цього порогу
    money_share = 1 - df_sorted.loc[idx, 'cum_balance']
    concentration_metrics[label] = {'threshold': val, 'wealth_share': money_share}

# ==========================================
# 3. ВІЗУАЛІЗАЦІЯ (Дашборд)
# ==========================================
fig = plt.figure(figsize=(22, 12))
gs = fig.add_gridspec(2, 2)

# --- Графік 1: Логарифмічний розподіл (Вгорі ліворуч) ---
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(np.log1p(dataset['BALANCE']), bins=60, kde=True, ax=ax1, color="royalblue")
ax1.axvline(np.log1p(current_hard_threshold), color='red', linestyle='--', linewidth=2, label=f'Поточний (800k)')
ax1.axvline(np.log1p(p25_hnwi), color='green', linestyle='-', linewidth=2, label=f'25-й перцентиль існ. HNWI')
ax1.set_title('Логарифмічний розподіл BALANCE\n(пошук бімодальності)', fontsize=14, fontweight='bold')
ax1.set_xlabel('log(BALANCE + 1)')
ax1.legend()

# --- Графік 2: Перетин сегментів (Вгорі праворуч) ---
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=dataset, x='SEGMENT', y='BALANCE', ax=ax2, order=['CONS', 'PREM', 'HNWI', 'EMPL'])
ax2.set_yscale('log')
ax2.axhline(current_hard_threshold, color='red', linestyle='--', linewidth=2, label='Поточний (800k)')
ax2.axhline(p25_hnwi, color='green', linestyle='-', linewidth=2, label='Нижня межа ядра HNWI (25%)')
ax2.set_title('Зв\'язок існуючих сегментів та реальних залишків', fontsize=14, fontweight='bold')
ax2.set_ylabel('BALANCE (log scale)')
ax2.legend()

# --- Графік 3: Крива Лоренца (Внизу, на всю ширину) ---
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(df_sorted['cum_clients'], df_sorted['cum_balance'], color='darkorange', linewidth=3, label='Крива Лоренца (Концентрація капіталу)')
ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Лінія абсолютної рівності')

# Додаємо точки наших цільових сегментів на криву
for label, metrics in concentration_metrics.items():
    t_val = targets[label]
    cum_bal = df_sorted[df_sorted['cum_clients'] >= t_val]['cum_balance'].iloc[0]
    ax3.plot(t_val, cum_bal, marker='o', markersize=8, color='red')
    ax3.annotate(f"{label}\nПоріг: {metrics['threshold']:,.0f}\nЧастка грошей: {metrics['wealth_share']:.1%}", 
                 xy=(t_val, cum_bal), xytext=(t_val - 0.1, cum_bal + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

ax3.set_title('Аналіз концентрації капіталу: Хто тримає гроші банку?', fontsize=14, fontweight='bold')
ax3.set_xlabel('Кумулятивна частка клієнтів (від найбідніших до найбагатіших)')
ax3.set_ylabel('Кумулятивна частка сумарного балансу')
ax3.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax3.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax3.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 4. ТЕКСТОВИЙ ЗВІТ (Вивід в консоль)
# ==========================================
print("=" * 60)
print(" АНАЛІТИЧНИЙ ЗВІТ: ОПТИМІЗАЦІЯ ПОРОГУ ДЛЯ СЕГМЕНТУ HNWI")
print("=" * 60)
print(f"Поточний жорсткий поріг: {current_hard_threshold:,.0f}")
print("-" * 60)
print("1. Аналіз на основі існуючої розмітки (Overlap із SEGMENT == 'HNWI'):")
print(f"   - 10-й перцентиль існуючих HNWI: {p10_hnwi:,.0f} (відсікає 10% найнижчих балансів у сегменті)")
print(f"   - 25-й перцентиль існуючих HNWI: {p25_hnwi:,.0f} (нижня межа 'ядра' сегменту)")
print(f"   - Медіана існуючих HNWI:         {median_hnwi:,.0f}")
print("-" * 60)
print("2. Аналіз концентрації капіталу (Розподіл Парето):")
for label, metrics in concentration_metrics.items():
    print(f"   - {label} найбагатших клієнтів:")
    print(f"       Поріг входу: {metrics['threshold']:,.0f}")
    print(f"       Ця група генерує {metrics['wealth_share']:.1%} від усіх пасивів.")
print("=" * 60)