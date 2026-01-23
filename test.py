# ==========================================
# 6. Фінальна чистка та форматування (Clean & Reorder)
# ==========================================

# 1. Список колонок, які треба видалити (проміжні або старі)
cols_to_drop = ['DESCR_NORM', 'Name', 'div_key', 'group_key', 'KVED_DESC'] 
# Додав 'KVED_DESC' про всяк випадок, якщо він там був. 
# errors='ignore' дозволить не впасти коду, якщо якоїсь колонки вже немає.
final_df = final_df.drop(columns=cols_to_drop, errors='ignore')

# 2. Словник для перейменування в CAPS
rename_map = {
    'Section_Code': 'SECTION_CODE',
    'Section_Name': 'SECTION_NAME',
    'Division_Code': 'DIVISION_CODE',
    'Division_Name': 'DIVISION_NAME',
    'Group_Code': 'GROUP_CODE',
    'Group_Name': 'GROUP_NAME',
    'Class_Name': 'KVED_NAME',  # Або 'CLASS_NAME', як вам зручніше
    'KVED': 'KVED_CODE',        # Можна залишити просто 'KVED'
    'KVED_Name': 'KVED_NAME'    # На випадок, якщо колонка називалась так
}

final_df = final_df.rename(columns=rename_map)

# 3. Визначаємо бажаний порядок колонок на початку (Ієрархія)
front_columns = [
    'SECTION_CODE', 'SECTION_NAME',
    'DIVISION_CODE', 'DIVISION_NAME',
    'GROUP_CODE', 'GROUP_NAME',
    'KVED', 'KVED_NAME' # Якщо ви перейменували KVED в KVED_CODE, замініть тут
]

# Фільтруємо список front_columns, залишаючи тільки ті, що реально є в датафреймі
# (на випадок, якщо якихось рівнів не знайшлося або назви трохи інші)
existing_front_cols = [c for c in front_columns if c in final_df.columns]

# 4. Формуємо решту колонок (все, що не входить в нашу ієрархію - EDRPOU, FIRM_NAME і т.д.)
other_columns = [c for c in final_df.columns if c not in existing_front_cols]

# 5. Перезбираємо DataFrame у правильному порядку
final_df = final_df[existing_front_cols + other_columns]

# Результат
print("Фінальний вигляд таблиці:")
display(final_df.head())