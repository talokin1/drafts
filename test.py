import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Дані
data = {
    'Month': ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    'Taken': [12670, 15927, 4064, 2720, 3029, 2933, 2244, 2373, 2263, 2004],
    'Acquired': [116, 149, 34, 18, 22, 28, 6, 8, 3, 9],
    'Conv_Rate': [0.0092, 0.0094, 0.0084, 0.0066, 0.0073, 0.0095, 0.0027, 0.0034, 0.0013, 0.0045],
    'Prop_Acq': [0.5384, 0.4906, 0.3449, 0.4447, 0.4095, 0.4250, 0.3546, 0.3468, 0.6045, 0.3093],
    'Prop_Not_Acq': [0.5745, 0.5465, 0.4014, 0.4429, 0.4301, 0.4116, 0.4184, 0.4080, 0.3887, 0.3807]
}
df = pd.DataFrame(data)

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Кольори
color_bar = '#4F81BD'
color_line = '#C0504D'
color_success = '#9BBB59'
color_fail = '#808080'

# ==========================================
# ГРАФІК 1: Об'єми та Конверсія (Окремий файл)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(6.5, 4.5), dpi=150) # Компактний розмір для 1/4 слайду

ax1.bar(df['Month'], df['Taken'], color=color_bar, alpha=0.75, label='Clients in Work')
ax1.set_ylabel('Number of Clients', color=color_bar, fontsize=10, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_bar, labelsize=9)
ax1.tick_params(axis='x', labelsize=9)
ax1.set_title('Processing Volumes & Conversion Rate Dynamics', fontsize=12, fontweight='bold', pad=10)

ax2 = ax1.twinx()
ax2.plot(df['Month'], df['Conv_Rate'], color=color_line, marker='o', linewidth=2, markersize=6, label='Conversion Rate')
ax2.set_ylabel('Conversion Rate', color=color_line, fontsize=10, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_line, labelsize=9)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

for i, txt in enumerate(df['Conv_Rate']):
    ax2.annotate(f"{txt:.2%}", (df['Month'][i], df['Conv_Rate'][i]), 
                 textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color=color_line, fontweight='bold')

ax1.grid(False)
ax2.grid(False)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, fontsize=8)

plt.tight_layout()
fig1.savefig('chart_1_volumes.png', transparent=True, bbox_inches='tight') # Зберігаємо з прозорим фоном
plt.close(fig1)

# ==========================================
# ГРАФІК 2: Дискримінаційна здатність (Окремий файл)
# ==========================================
fig2, ax3 = plt.subplots(figsize=(6.5, 4.5), dpi=150)

ax3.plot(df['Month'], df['Prop_Acq'], marker='s', color=color_success, linewidth=2, markersize=6, label='Propensity (Acquired)')
ax3.plot(df['Month'], df['Prop_Not_Acq'], marker='^', color=color_fail, linewidth=2, linestyle='--', markersize=6, label='Propensity (Not Acquired)')

ax3.set_title('Model Discrimination Ability', fontsize=12, fontweight='bold', pad=10)
ax3.set_ylabel('Average Propensity Score', fontsize=10)
ax3.tick_params(axis='both', labelsize=9)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax3.legend(loc='upper right', frameon=True, fontsize=8)

ax3.fill_between(df['Month'], df['Prop_Acq'], df['Prop_Not_Acq'], 
                 where=(df['Prop_Acq'] >= df['Prop_Not_Acq']), interpolate=True, color=color_success, alpha=0.15)
ax3.fill_between(df['Month'], df['Prop_Acq'], df['Prop_Not_Acq'], 
                 where=(df['Prop_Acq'] < df['Prop_Not_Acq']), interpolate=True, color='#C0504D', alpha=0.1)

plt.tight_layout()
fig2.savefig('chart_2_model.png', transparent=True, bbox_inches='tight')
plt.close(fig2)

print("Два окремі графіки збережено!")