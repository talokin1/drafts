import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Створюємо змінну BALANCE (як у твоєму коді)
dataset["BALANCE"] = (dataset["AMOUNT_SAVINGS"] + 
                      dataset["BALANCE_CURRENT_ACCOUNTS"] + 
                      dataset["BALANCE_DEBIT_CARDS"] + 
                      dataset["BALANCE_SALARY_CARDS"] + 
                      dataset["BALANCE_SOCIAL_CARDS"])

# Налаштування стилю
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Графік 1: Гістограма з логарифмічною шкалою X ---
# Використовуємо log1p, щоб уникнути помилки log(0) для клієнтів з нульовим балансом
sns.histplot(np.log1p(dataset['BALANCE']), bins=50, kde=True, ax=axes[0], color="royalblue")
axes[0].axvline(np.log1p(800000), color='red', linestyle='--', label='Поточний поріг (800k)')
axes[0].set_title('Логарифмічний розподіл BALANCE\n(log1p масштаб)')
axes[0].set_xlabel('log(BALANCE + 1)')
axes[0].legend()

# --- Графік 2: ECDF (Емпірична функція розподілу) ---
sns.ecdfplot(data=dataset, x='BALANCE', ax=axes[1], color="darkorange")
axes[1].axvline(800000, color='red', linestyle='--', label='Поточний поріг (800k)')
axes[1].set_xscale('log') # Логарифмуємо тільки вісь X для наочності
axes[1].set_title('ECDF: Частка клієнтів нижче порогу')
axes[1].set_ylabel('Частка клієнтів (Cumulative Probability)')
axes[1].legend()

# --- Графік 3: Зв'язок балансу з існуючим сегментом ---
sns.boxplot(data=dataset, x='SEGMENT', y='BALANCE', ax=axes[2], palette="Set2")
axes[2].set_yscale('log')
axes[2].axhline(800000, color='red', linestyle='--', label='Поточний поріг (800k)')
axes[2].set_title('Розподіл BALANCE в розрізі існуючих сегментів')
axes[2].legend()

plt.tight_layout()
plt.show()


