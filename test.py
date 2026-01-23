def normalize_kved(val):
    if pd.isna(val):
        return val
    
    # 1. Заміна коми та видалення зайвих пробілів
    val_str = str(val).replace(',', '.').strip()
    
    # 2. Розбиваємо на частини "до крапки" і "після"
    parts = val_str.split('.')
    
    # Частина 1 (Основна): Додаємо нуль зліва, якщо треба (1 -> 01)
    major = parts[0].zfill(2)
    
    # Частина 2 (Дробова): Додаємо нуль СПРАВА, якщо треба (1 -> 10)
    if len(parts) > 1:
        # ljust(2, '0') доповнює рядок нулями справа до 2 символів
        minor = parts[1].ljust(2, '0')
        return f"{major}.{minor}"
    else:
        # Якщо крапки не було (наприклад, код "72")
        # Якщо ви хочете перетворити "72" на "72.00", розкоментуйте рядок нижче:
        # return f"{major}.00"
        return major

# Застосовуємо до колонки
fin_ind['KVED'] = fin_ind['KVED'].apply(normalize_kved)

# Перевіряємо результат (тепер 81.1 та 81.10 зіллються в 81.10)
print(fin_ind['KVED'].value_counts().head(10))