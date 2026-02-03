# 1. Знаходимо "шкідливе" значення (найпопулярніше > 0)
# Працюємо з df_clean, де вже прибрані "кити" (мільйонери)
target_col = "CURR_ACC"

# Перевірка: чи таргет вже логарифмований?
# Якщо максимум < 20, то швидше за все так.
is_logged = df[target_col].max() < 20
print(f"Is target already logged? {is_logged}")

# Знаходимо моду (пік)
spike_value = df[target_col].mode()[0]
print(f"Detected Spike Value (Log scale): {spike_value:.4f}")
print(f"Real Money Equivalent: {np.expm1(spike_value):.2f}") 

# 2. Фільтрація (Вирізаємо "голку")
# Якщо це технічний залишок, він зазвичай має дуже вузький діапазон
# Видалимо все, що дуже близько до піку (наприклад, +/- 0.01)
df_final = df[~((df[target_col] >= spike_value - 0.01) & 
                (df[target_col] <= spike_value + 0.01))].copy()

print(f"Rows dropped: {len(df) - len(df_final)}")

# 3. Правильний графік (без подвійного логарифма)
plt.figure(figsize=(10, 5))
# Увага: тут ми просто малюємо df_final[target_col], бо він ВЖЕ логарифмований
sns.histplot(df_final[target_col], bins=100)
plt.title("Clean Distribution (No Spike, No Whales)")
plt.show()