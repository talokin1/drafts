import pandas as pd
import numpy as np

# 1. Функція для математичного відновлення формату
def format_to_xxxx(val):
    # Перетворюємо на рядок і забираємо зайві пробіли
    val = str(val).strip()

    # Відкидаємо пусті значення
    if val in ['Missing value', 'nan', '', 'None']:
        return np.nan

    # Математична логіка форматування за довжиною рядка:
    if len(val) == 4:
        # Приклад: '9700' -> '97.00'
        return f"{val[:2]}.{val[2:]}"

    elif len(val) == 3:
        # Приклад: '111' -> '01.11', '710' -> '07.10'
        # Додаємо втрачений нуль на початок
        val = '0' + val
        return f"{val[:2]}.{val[2:]}"

    elif len(val) == 2:
        # Приклад: '97' -> '97.00'
        return f"{val}.00"

    elif len(val) == 1:
        # Приклад: '1' -> '01.00'
        return f"0{val}.00"

    return val

# 2. Створюємо нашу таблицю з двома колонками
kved_colors = kved[['KVED', 'Risk classification - Jan 2025']].copy()
kved_colors = kved_colors.rename(columns={'Risk classification - Jan 2025': 'RISK_COLOR'})

# 3. Застосовуємо форматування
kved_colors['KVED'] = kved_colors['KVED'].apply(format_to_xxxx)

# 4. Видаляємо рядки, де КВЕД відсутній
kved_colors = kved_colors.dropna(subset=['KVED'])

print(kved_colors.head(10))