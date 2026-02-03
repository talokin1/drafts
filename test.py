import pandas as pd
import numpy as np

# 1. РОБИМО ПРЕДИКТ (Original Log Scale)
y_pred_log = reg.predict(X_val)

# 2. РОЗТЯГУВАННЯ (Variance Inflation)
# Це критичний крок для "Потенціалу"
std_true = np.std(y_val_log)
std_pred = np.std(y_pred_log)
mean_true = np.mean(y_val_log)
mean_pred = np.mean(y_pred_log)

scaling_factor = std_true / std_pred
y_pred_stretched = (y_pred_log - mean_pred) * scaling_factor + mean_true

# 3. ЗБИРАЄМО РЕЗУЛЬТАТИ В DATAFRAME
df_res = pd.DataFrame({
    'True_Money': np.expm1(y_val_log),       # Факт (грн)
    'Pred_Potential': np.expm1(y_pred_stretched), # Прогноз Потенціалу (грн)
    'Pred_Original': np.expm1(y_pred_log)    # Обережний прогноз (для порівняння)
})

# Розбиваємо на 10 груп за прогнозом Потенціалу
df_res['Decile'] = pd.qcut(df_res['Pred_Potential'], 10, labels=False) + 1

# 4. БУДУЄМО ТАБЛИЦЮ (з Медіаною!)
pd.options.display.float_format = '{:,.0f}'.format

decile_stats = df_res.groupby('Decile')['True_Money'].agg(
    Min='min', 
    Median='median',  # <--- Головна метрика для перевірки
    Mean='mean', 
    Max='max', 
    Count='count'
)

print("=== DECILE ANALYSIS (Median vs Mean) ===")
print(decile_stats.sort_index(ascending=False))
print("-" * 60)

# 5. АНАЛІЗ ПОМИЛОК ("Полювання на китів")
# Давай знайдемо того самого мільйонера, який зіпсував статистику в 4-му децилі
# Шукаємо клієнтів з низьким децилем (<=5), але купою грошей (> 100 000)

whales = df_res[
    (df_res['Decile'] <= 5) & 
    (df_res['True_Money'] > 100_000)
].sort_values('True_Money', ascending=False)

print(f"\nЗнайдено {len(whales)} 'китів', яких модель пропустила (Low Potential, High Actual):")
if len(whales) > 0:
    print(whales.head(5)) # Топ-5 помилок
else:
    print("Супер! Грубих помилок немає.")