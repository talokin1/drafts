def fix_kved_format(kved_code):
    # Перетворюємо на рядок і чистимо пробіли
    code_str = str(kved_code).strip()
    
    # Розбиваємо код по крапці
    parts = code_str.split('.')
    
    # Беремо першу частину (розділ) і додаємо нуль зліва, якщо там 1 цифра
    # '1' -> '01', '46' -> '46'
    parts[0] = parts[0].zfill(2)
    
    # Збираємо назад
    return ".".join(parts)


companies_df['KVED'] = companies_df['KVED'].apply(fix_kved_format)