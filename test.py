import pandas as pd
import json

# ==========================================
# 1. Підготовка довідників з JSON
# ==========================================

# Завантажуємо JSON
with open('kved.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Функція очищення (прибираємо \n)
def clean_record(record):
    return {k.strip(): v for k, v in record.items()}

data = [clean_record(row) for row in raw_data]

# Створюємо словники для кожного рівня ієрархії
sections_name_map = {}   # 'A' -> 'Сільське господарство...'
divisions_name_map = {}  # '01' -> 'Сільське господарство...'
groups_name_map = {}     # '01.1' -> 'Вирощування...'
classes_name_map = {}    # '01.11' -> 'Вирощування зернових...'

# Додатковий словник: щоб знати, до якої Секції (літери) належить Розділ (цифри)
# Наприклад: '01' -> 'A'
division_to_section_code = {}

for row in data:
    sec_code = row.get('Код секції')
    div_code = row.get('Код розділу')
    grp_code = row.get('Код групи')
    cls_code = row.get('Код класу')
    name = row.get('Назва')

    # Заповнюємо довідник Секцій
    if sec_code and not div_code:
        sections_name_map[sec_code] = name

    # Заповнюємо довідник Розділів і зв'язок з Секцією
    if div_code:
        # Якщо це запис самого розділу (без групи)
        if not grp_code:
            divisions_name_map[div_code] = name
        # Запам'ятовуємо, що цей розділ належить цій секції
        if sec_code:
            division_to_section_code[div_code] = sec_code

    # Заповнюємо довідник Груп
    if grp_code and not cls_code:
        groups_name_map[grp_code] = name

    # Заповнюємо довідник Класів
    if cls_code:
        classes_name_map[cls_code] = name

# ==========================================
# 2. Обробка вашої таблиці (Enrichment)
# ==========================================

# Припустимо, це ваш DataFrame
# companies_df = pd.read_csv(...) 
# Для прикладу створимо df з проблемними кодами
# companies_df = pd.DataFrame({'KVED': ['91.31', '74.13', '51.11', '01.11']})

# 1. Переконуємося, що KVED - це рядок
companies_df['KVED'] = companies_df['KVED'].astype(str).str.strip()

# 2. Витягуємо компоненти коду з колонки KVED
# Розділ - це завжди перші 2 символи (наприклад '91' з '91.31')
companies_df['div_key'] = companies_df['KVED'].str[:2]

# Група - це зазвичай перші 4 символи (наприклад '91.3' з '91.31')
# Але треба бути обережним, якщо код короткий. Беремо до 4 символів.
companies_df['group_key'] = companies_df['KVED'].str[:4]

# 3. Мапимо назви (Lookup)
# Шукаємо назву класу (точне співпадіння)
companies_df['Class_Name'] = companies_df['KVED'].map(classes_name_map)

# Шукаємо назву групи (за ключем 'XX.X')
companies_df['Group_Name'] = companies_df['group_key'].map(groups_name_map)

# Шукаємо назву розділу (за ключем 'XX')
companies_df['Division_Name'] = companies_df['div_key'].map(divisions_name_map)

# 4. Визначаємо Секцію (Літеру і Назву) через Розділ
# Спочатку знаходимо код секції (A, B, C...) через код розділу
companies_df['Section_Code'] = companies_df['div_key'].map(division_to_section_code)
# Тепер знаходимо назву секції
companies_df['Section_Name'] = companies_df['Section_Code'].map(sections_name_map)

# ==========================================
# 5. Фінальна чистка
# ==========================================

# Видаляємо допоміжні ключі, якщо вони не потрібні
final_df = companies_df.drop(columns=['div_key', 'group_key'])

print("Результат обробки:")
display(final_df.head())

# Перевірка: скільки записів отримали хоча б назву Секції або Розділу
filled_stats = final_df[['Section_Name', 'Division_Name', 'Group_Name', 'Class_Name']].notna().sum()
print("\nСтатистика заповнення:")
print(filled_stats)