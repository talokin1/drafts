def clean_opf_code_final(val):
    # Якщо значення пусте, повертаємо як є
    if pd.isna(val) or val == "":
        return val
    
    try:
        # 1. Конвертуємо через float в int, щоб гарантовано прибрати ".0"
        # Це працює і для числа 220.0, і для рядка "220.0"
        int_val = int(float(val))
        
        # 2. Якщо це 0 (який часто означає "немає даних"),
        # ви можете або залишити його '0000', або повернути None.
        # Зараз робимо 0000 згідно з вашим правилом "4 цифри":
        
        # 3. Перетворюємо в рядок і додаємо нулі СПРАВА (ljust)
        return str(int_val).ljust(4, '0')
        
    except (ValueError, TypeError):
        # Якщо там якесь текстове сміття, повертаємо як є
        return val

# Застосовуємо до колонки
final_df['OPF_CODE'] = final_df['OPF_CODE'].apply(clean_opf_code_final)

# Перевіряємо результат
print(final_df['OPF_CODE'].value_counts().head(10))