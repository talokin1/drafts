# 1. Знаходимо поріг на оригінальних даних
q99 = df[TARGET_NAME].quantile(0.99)

# 2. Робимо копію, щоб не зламати оригінал
df_ = df.copy()

# 3. ВІНЗОРИЗАЦІЯ: обмежуємо зверху, але НЕ видаляємо рядки!
df_[TARGET_NAME] = df_[TARGET_NAME].clip(upper=q99)

# 4. Вже тепер логарифмуємо
df_["LOG_TARGET"] = np.log1p(df_[TARGET_NAME])

# 5. Відкидаємо нулі або дрібний шум (це залишаємо, як у тебе)
df_ = df_[df_["LOG_TARGET"] >= 1]

# Дивимось на розподіл
plt.figure(figsize=(8,6))
sns.histplot(df_["LOG_TARGET"], bins=200)
plt.title("Winsorized & Logged Target")
plt.show()