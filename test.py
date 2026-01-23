def clean_kved(val):
    # Якщо значення пусте, повертаємо як є
    if pd.isna(val):
        return val
    
    # 1. Конвертуємо в рядок і замінюємо кому на крапку
    val_str = str(val).strip().replace(',', '.')
    
    # 2. Розбиваємо по крапці
    parts = val_str.split('.')
    
    # 3. Якщо перша частина (до крапки) має менше 2 цифр, додаємо нуль зліва (1 -> 01)
    if len(parts) > 0:
        parts[0] = parts[0].zfill(2)
        
    # Збираємо назад через крапку
    return ".".join(parts)

# Застосовуємо до колонки
fin_ind['KVED'] = fin_ind['KVED'].apply(clean_kved)

# Перевірка результату
print(fin_ind['KVED'].unique())