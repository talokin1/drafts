import pandas as pd
import glob
import os

# Вкажи шлях до папки, куди ти щойно зберіг усі Excel-файли
input_folder = r"C:\Projects\Alina Kinash\data_for_dashboard\raw_potential_clients"
# Шлях, куди ми збережемо фінальний чистий файл (який потім піде в основний скрипт)
output_file = r"C:\Projects\Alina Kinash\data_for_dashboard\potential_clients.csv"

print("Починаю збір даних з Excel-файлів...")

# Шукаємо всі файли з розширенням .xlsx
all_files = glob.glob(os.path.join(input_folder, "*.xlsx"))
df_list = []

print(f"Знайдено файлів: {len(all_files)}")

for file in all_files:
    # Читаємо файл
    df_temp = pd.read_excel(file)
    
    # Витягуємо назву файлу, щоб знати, звідки клієнт
    file_name = os.path.basename(file).replace('.xlsx', '')
    
    # Динамічно шукаємо колонку з ID (ігноруємо регістр)
    id_col = None
    for col in df_temp.columns:
        if str(col).strip().upper() in ['IDENTIFYCODE', 'CONTRAGENTID', 'ID', 'ЄДРПОУ', 'ІПН', 'IDENTIFY_CODE']:
            id_col = col
            break
            
    # Якщо специфічної назви немає, беремо першу колонку як ключ
    if id_col is None:
        id_col = df_temp.columns[0]
        
    # Створюємо підмножину: тільки ID та джерело
    temp_subset = df_temp[[id_col]].copy()
    temp_subset.rename(columns={id_col: 'IDENTIFYCODE'}, inplace=True)
    
    # Жорстка типізація: перетворюємо ID на текст, видаляємо пробіли та ".0"
    temp_subset['IDENTIFYCODE'] = temp_subset['IDENTIFYCODE'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    # Додаємо колонку з назвою банку (на основі назви файлу)
    # Відрізаємо зайве, наприклад "UkRExim_clients Дніпро" -> "UkRExim"
    bank_name = file_name.split('_')[0].capitalize() 
    if "MOTOR" in file_name.upper(): bank_name = "Motor Bank"
    elif "EXIM" in file_name.upper() or "ЕКСІМ" in file_name.upper(): bank_name = "Ukreximbank"
    
    temp_subset['BANK'] = bank_name
    
    df_list.append(temp_subset)

# 1. Об'єднуємо всі файли в одну велику матрицю (Union)
df_all = pd.concat(df_list, ignore_index=True)
print(f"Загальна кількість рядків до очистки: {len(df_all)}")

# 2. Видаляємо дублікати (якщо один клієнт є в кількох файлах/регіонах)
# keep='first' означає, що ми залишимо першу згадку про клієнта
df_unique = df_all.drop_duplicates(subset=['IDENTIFYCODE'], keep='first')

# 3. Зберігаємо результат
df_unique.to_csv(output_file, index=False)

print(f"ГОТОВО! Потужність унікальної множини |P| (Потенційні клієнти): {len(df_unique)}")