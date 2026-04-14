import pandas as pd
import numpy as np

# 1. Створюємо списки колонок за їхнім логічним типом
bool_like_cols = ['is_abroad', 'has_report', 'auction_possible', 'exchange_possible']
numeric_cols = ['doors_count'] # Сюди можна додати інші числові, якщо там теж є "Missing value"

# 2. Надійний мапінг булевих колонок
for col in bool_like_cols:
    # Приводимо до рядка, прибираємо пробіли, робимо нижній регістр для надійності
    clean_series = df[col].astype(str).str.strip().str.lower()
    
    # Все, що не збігається з ключами словника, автоматично стане NaN
    df[col] = clean_series.map({
        'true': 1,
        '1.0': 1,
        '1': 1,
        'false': 0,
        '0.0': 0,
        '0': 0
    })

# 3. Конвертація суто числових колонок (замінюємо текст на NaN, потім у float)
for col in numeric_cols:
    # Спочатку безпечно замінюємо конкретний текст, якщо він є
    df[col] = df[col].replace('Missing value', np.nan)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Перевірка
print("Пропуски після обробки:\n", df[['is_abroad', 'doors_count']].isna().sum())
print("\nТипи даних:\n", df[['is_abroad', 'doors_count']].dtypes)