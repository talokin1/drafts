import pandas as pd
import json

# ==========================================
# 1. Завантаження та очищення JSON
# ==========================================

# Замініть шлях на ваш файл
file_path = 'kved.json' 

with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Функція для видалення \n та пробілів з ключів
def clean_record(record):
    return {k.strip(): v for k, v in record.items()}

# Очищуємо всі записи
data = [clean_record(row) for row in raw_data]

# ==========================================
# 2. Створення довідників (Lookups)
# ==========================================
# JSON містить окремі об'єкти для опису Секцій, Розділів та Груп.
# Нам треба витягнути їх назви, щоб потім підставити до Класів.

sections_map = {}   # 'C' -> 'Переробна промисловість'
divisions_map = {}  # '23' -> 'Виробництво іншої неметалевої...'
groups_map = {}     # '23.1' -> 'Виробництво скла та виробів зі скла'
kved_classes = []   # Сюди збережемо тільки кінцеві класи (23.11)

for row in data:
    # Логіка визначення рівня ієрархії за наявністю ключів
    
    # Це Клас (кінцевий рівень, який нам треба в таблиці)
    if 'Код класу' in row:
        kved_classes.append(row)
        
    # Це Група (наприклад 23.1) - зберігаємо назву
    elif 'Код групи' in row:
        groups_map[row['Код групи']] = row['Назва']
        
    # Це Розділ (наприклад 23) - зберігаємо назву
    elif 'Код розділу' in row:
        divisions_map[row['Код розділу']] = row['Назва']
        
    # Це Секція (наприклад C) - зберігаємо назву
    elif 'Код секції' in row:
        sections_map[row['Код секції']] = row['Назва']

# ==========================================
# 3. Збірка майстер-таблиці КВЕД
# ==========================================

final_rows = []

for item in kved_classes:
    # Беремо коди з поточного запису класу
    sec_code = item.get('Код секції')
    div_code = item.get('Код розділу')
    grp_code = item.get('Код групи')
    class_code = item.get('Код класу')
    
    final_rows.append({
        'KVED': class_code,  # Це поле буде ключем для мерджу
        'KVED_Name': item['Назва'], # Назва самого класу (Виробництво листового скла)
        
        # Підтягуємо назви батьківських рівнів зі словників
        'Group_Code': grp_code,
        'Group_Name': groups_map.get(grp_code, ''),
        
        'Division_Code': div_code,
        'Division_Name': divisions_map.get(div_code, ''),
        
        'Section_Code': sec_code,
        'Section_Name': sections_map.get(sec_code, '')
    })

kved_reference_df = pd.DataFrame(final_rows)

# ==========================================
# 4. Мердж з вашою таблицею компаній
# ==========================================

# Припустимо, df_companies - це ваша існуюча таблиця
# df_companies = pd.read_csv(...) або вже є в пам'яті

# ВАЖЛИВО: Переконайтесь, що типи даних однакові (string)
# Якщо в df_companies колонка називається інакше, змініть 'KVED' на вашу назву
# kved_reference_df['KVED'] = kved_reference_df['KVED'].astype(str)

# df_companies = df_companies.merge(
#     kved_reference_df, 
#     on='KVED', 
#     how='left'
# )

# Результат:
print("Приклад сформованого довідника:")
display(kved_reference_df.head())