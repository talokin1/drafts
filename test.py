import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_ = df.copy()

# 1. Спочатку відкидаємо тих, хто не має кредитів (або має мікрозалишки)
# У твоєму коді це було еквівалентно log1p(x) >= 1, тобто x >= e^1 - 1 (приблизно 1.71)
df_active = df_[df_[TARGET_NAME] > 2].copy() # Відсіюємо нулі

# 2. Рахуємо 99-й перцентиль ТІЛЬКИ по активних позичальниках
q99 = df_active[TARGET_NAME].quantile(0.99)

# 3. Вінзоризація
df_active[TARGET_NAME] = df_active[TARGET_NAME].clip(upper=q99)

# 4. Логарифмування
df_active["LOG_TARGET"] = np.log1p(df_active[TARGET_NAME])

# Дивимось на результат
plt.figure(figsize=(8,6))
sns.histplot(df_active["LOG_TARGET"], bins=200)
plt.title("Winsorized & Logged Target (Active Portfolio Only)")
plt.show()