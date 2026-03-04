import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Дані з твоїх скріншотів для побудови графіків
data = {
    'Month': ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    'Taken': [12670, 15927, 4064, 2720, 3029, 2933, 2244, 2373, 2263, 2004],
    'Acquired': [116, 149, 34, 18, 22, 28, 6, 8, 3, 9],
    'Conv_Rate': [0.0092, 0.0094, 0.0084, 0.0066, 0.0073, 0.0095, 0.0027, 0.0034, 0.0013, 0.0045],
    'Prop_Acq': [0.5384, 0.4906, 0.3449, 0.4447, 0.4095, 0.4250, 0.3546, 0.3468, 0.6045, 0.3093],
    'Prop_Not_Acq': [0.5745, 0.5465, 0.4014, 0.4429, 0.4301, 0.4116, 0.4184, 0.4080, 0.3887, 0.3807]
}
df = pd.DataFrame(data)

# Налаштування стилю (робимо чисто та корпоративно)
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial'] # Корпоративні шрифти

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), dpi=150)
fig.tight_layout(pad=6.0)

# ==========================================
# Графік 1: Воронка обробки та Конверсія (Комбінований)
# ==========================================
color_bar = '#4F81BD' # Спокійний синій
color_line = '#C0504D' # Акцентний червоний

# Стовпчики: взяті в роботу
ax1.bar(df['Month'], df['Taken'], color=color_bar, alpha=0.7, label='Клієнти в роботі (W)')
ax1.set_ylabel('Кількість клієнтів', color=color_bar, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_bar)
ax1.set_title('Об\'єми опрацювання та динаміка Конверсії', fontsize=14, fontweight='bold', pad=15)

# Лінія: Конверсія (на додатковій осі Y)
ax2 = ax1.twinx()
ax2.plot(df['Month'], df['Conv_Rate'], color=color_line, marker='o', linewidth=2.5, markersize=8, label='Конверсія (CR)')
ax2.set_ylabel('Конверсія', color=color_line, fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_line)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

# Додаємо підписи значень на лінію конверсії
for i, txt in enumerate(df['Conv_Rate']):
    ax2.annotate(f"{txt:.2%}", (df['Month'][i], df['Conv_Rate'][i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color=color_line, fontweight='bold')

ax1.grid(False)
ax2.grid(False)

# ==========================================
# Графік 2: Дискримінаційна здатність моделі
# ==========================================
color_success = '#9BBB59' # Зелений для успіху
color_fail = '#808080'    # Сірий для неуспіху

ax3.plot(df['Month'], df['Prop_Acq'], marker='s', color=color_success, linewidth=2.5, markersize=8, label='Схильність залучених E[P|S]')
ax3.plot(df['Month'], df['Prop_Not_Acq'], marker='^', color=color_fail, linewidth=2.5, linestyle='--', markersize=8, label='Схильність незалучених E[P|~S]')

ax3.set_title('Дискримінаційна здатність моделі (Propensity Score)', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylabel('Середній % схильності (Primary)', fontsize=11)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax3.legend(loc='upper right', frameon=True, fontsize=10)

# Додаємо заливку між лініями для інтуїтивного розуміння, де модель перемагає
ax3.fill_between(df['Month'], df['Prop_Acq'], df['Prop_Not_Acq'], 
                 where=(df['Prop_Acq'] >= df['Prop_Not_Acq']), interpolate=True, color=color_success, alpha=0.2)
ax3.fill_between(df['Month'], df['Prop_Acq'], df['Prop_Not_Acq'], 
                 where=(df['Prop_Acq'] < df['Prop_Not_Acq']), interpolate=True, color='red', alpha=0.1)

# Зберігаємо результат
plt.savefig('business_slide_charts.png', bbox_inches='tight')
plt.show()