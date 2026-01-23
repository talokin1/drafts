# 1. Нормалізація регістру в обох таблицях (приводимо до нижнього)
# Використовуємо .astype(str), щоб уникнути помилок, але треба обережно з NaN
final_df['FIRM_OPFNM'] = final_df['FIRM_OPFNM'].str.lower().str.strip()
ubki['OPF'] = ubki['OPF'].str.lower().str.strip()

# 2. Підготовка даних з UBKI
# Перейменуємо колонки, щоб при мерджі було зрозуміло, де дані з UBKI
ubki_subset = ubki[['IDENTIFYCODE', 'OPF_CODE', 'OPF']].rename(columns={
    'OPF_CODE': 'UBKI_OPF_CODE', 
    'OPF': 'UBKI_OPF_NAME'
})

# 3. Виконуємо Merge
final_df = final_df.merge(ubki_subset, on='IDENTIFYCODE', how='left')

# 4. Логіка заповнення (як з КВЕД)
# Якщо в UBKI є код/назва — беремо їх. Якщо ні — залишаємо те, що було в final_df.

# Оновлюємо КОД (OPF_CODE)
final_df['FIRM_OPFCD'] = final_df['UBKI_OPF_CODE'].fillna(final_df['FIRM_OPFCD'])

# Оновлюємо НАЗВУ (OPF_NAME)
final_df['FIRM_OPFNM'] = final_df['UBKI_OPF_NAME'].fillna(final_df['FIRM_OPFNM'])

# 5. Повертаємо велику літеру (Capitalize)
# "товариство з обмеженою..." -> "Товариство з обмеженою..."
final_df['FIRM_OPFNM'] = final_df['FIRM_OPFNM'].str.capitalize()

# 6. Видаляємо тимчасові колонки з UBKI
final_df = final_df.drop(columns=['UBKI_OPF_CODE', 'UBKI_OPF_NAME'])

# Перевірка результату
print(final_df[['IDENTIFYCODE', 'FIRM_OPFCD', 'FIRM_OPFNM']].head())