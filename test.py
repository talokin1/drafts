import numpy as np
import pandas as pd

# 1. Відбираємо тільки тих, кого ми вважаємо "цільовими" (не нульові)
# Бо ми оцінюємо точність саме для них
vip_df = df[df['CURR_ACC'] > 1000] # Або ваш поріг з класифікатора

mean_val = vip_df['CURR_ACC'].mean()
median_val = vip_df['CURR_ACC'].median()
std_val = vip_df['CURR_ACC'].std()

print(f"--- Статистика реальних даних (VIP сегмент) ---")
print(f"Середнє (Mean): {mean_val:,.2f} грн")
print(f"Медіана (Median): {median_val:,.2f} грн")
print(f"Станд. відхилення (Std): {std_val:,.2f} грн")
print(f"-"*30)
print(f"Ваша помилка (MAE): 8975.00 грн")

# Відносна оцінка
print(f"Помилка у % від Середнього: {(8975 / mean_val) * 100:.2f}%")
print(f"Помилка у % від Медіани:    {(8975 / median_val) * 100:.2f}%")








# Рахуємо відсоткову помилку для кожного клієнта окремо
# Додаємо epsilon=1e-6, щоб не ділити на нуль
errors_percentage = np.abs((y_test_real - final_preds) / (y_test_real + 1e-6)) * 100

# Беремо медіану відсоткових помилок (бо середнє MAPE може злетіти через один дрібний рахунок)
median_mape = np.median(errors_percentage)

print(f"Медіанна відсоткова помилка (MdAPE): {median_mape:.2f}%")








import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
# Малюємо розподіл реальних грошей (обрізаємо екстремальні викиди для краси)
sns.histplot(vip_df['CURR_ACC'], bins=50, log_scale=True, color='blue', label='Реальні гроші')
# Малюємо лінію помилки
plt.axvline(x=8975, color='red', linestyle='--', linewidth=2, label='Ваша MAE (8975 грн)')

plt.title('Де ваша помилка відносно грошей клієнтів?')
plt.xlabel('Сума на рахунку (Log Scale)')
plt.legend()
plt.show()