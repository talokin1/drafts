# 1. АНАЛІЗ: Що це за дивна "голка" на графіку?
# Дивимось найпопулярніші значення (моди)
print("Top 5 most frequent values (Check for technical spikes):")
print(df["CURR_ACC"].value_counts().head(5))

# 2. АНАЛІЗ: Хто ці "кити" справа?
# Дивимось 99.9-й перцентиль (топ 0.1% найбагатших)
q_high = df["CURR_ACC"].quantile(0.999)
print(f"\n99.9% cutoff value: {q_high:.2f}")
print(f"Rows above cutoff: {(df['CURR_ACC'] > q_high).sum()}")

# -------------------------------------------------------
# 3. ЧИСТКА (Вибирай варіант А або Б)
# -------------------------------------------------------

# ВАРІАНТ А: Обрізати "хвіст" (найпопулярніший метод)
# Ми просто викидаємо топ 0.1% або 0.5% клієнтів, які псують масштаб
df_clean = df[df["CURR_ACC"] < q_high].copy()

# ВАРІАНТ Б: Sigma Clipping (статистичний метод)
# Викидаємо все, що далі 3-х стандартних відхилень (в логарифмічній шкалі)
# Це краще для Гауссового розподілу, який ми бачимо на картинці
log_data = np.log1p(df["CURR_ACC"])
mean = log_data.mean()
std = log_data.std()
lower_bound = mean - 3 * std
upper_bound = mean + 3 * std

# Фільтруємо за логарифмом
df_sigma = df[(log_data >= lower_bound) & (log_data <= upper_bound)].copy()

print(f"\nOriginal Size: {len(df)}")
print(f"After Quantile cut: {len(df_clean)}")
print(f"After Sigma cut: {len(df_sigma)}")

# 4. Перевірка результату (малюємо новий графік)
plt.figure(figsize=(10, 5))
sns.histplot(np.log1p(df_clean["CURR_ACC"]), bins=100)
plt.title("Distribution after removing outliers")
plt.show()