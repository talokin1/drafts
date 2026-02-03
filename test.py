import pandas as pd
import numpy as np

# 1. Готуємо дані (повертаємось з логарифмів у реальні гроші)
df_check = pd.DataFrame({
    'True_Money': np.expm1(y_val_log),       # Реальність
    'Pred_Potential': y_pred_stretched       # Твій прогноз (лог) для сортування
})

# 2. Розбиваємо на 10 груп (децилів) за Прогнозом
# qcut ділить на рівні групи
df_check['Decile'] = pd.qcut(df_check['Pred_Potential'], 10, labels=False) + 1

# 3. Агрегуємо: скільки РЕАЛЬНО грошей у кожній групі
decile_stats = df_check.groupby('Decile')['True_Money'].agg(['mean', 'min', 'max', 'count'])

# Додаємо красиве форматування
pd.options.display.float_format = '{:,.0f}'.format

print("=== DECILE ANALYSIS (Monotonicity Check) ===")
print("Decile 10 = Клієнти з найвищим прогнозом")
print("Mean True Money = Скільки у них реально грошей в середньому")
print("-" * 60)
print(decile_stats.sort_index(ascending=False)) # Сортуємо, щоб Топ був зверху
print("-" * 60)

# Перевірка на монотонність
means = decile_stats['mean'].tolist()
is_monotonic = all(x <= y for x, y in zip(means, means[1:]))
print(f"Чи росте реальний дохід разом з прогнозом? -> {'ТАК ✅' if is_monotonic else 'НІ ❌'}")