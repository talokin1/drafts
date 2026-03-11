import pandas as pd
import os
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Вимикаємо зайві попередження від Excel

folder_path = 'potential_clients/' # Вкажи свій шлях
zkp_file = 'ZKP_PROJECT_BY_MARCH.xlsx'

print("Починаю глибоку очистку та зведення даних...")

# ==========================================
# 1. Формуємо підмножину потенційних клієнтів (P)
# ==========================================
all_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
df_p_list = []

for file in all_files:
    df_temp = pd.read_excel(file)
    bank_name = os.path.basename(file).replace('_clients.xlsx', '').replace('.xlsx', '')
    
    # ФІКС 1: Динамічний пошук колонки з ID (ігноруємо регістр і пробіли)
    id_col_name = None
    for col in df_temp.columns:
        if str(col).strip().upper() in ['IDENTIFYCODE', 'CONTRAGENTID', 'ID', 'IDENTIFY_CODE']:
            id_col_name = col
            break
    
    # Якщо не знайшли за назвою, беремо першу колонку за замовчуванням
    if id_col_name is None:
        id_col_name = df_temp.columns[0]
        
    # Залишаємо тільки ID і приводимо назву до стандарту
    temp_subset = df_temp[[id_col_name]].copy()
    temp_subset.rename(columns={id_col_name: 'IDENTIFYCODE'}, inplace=True)
    
    # ФІКС 2: Жорстка типізація. Перетворюємо все на чистий текст, видаляємо ".0" і пробіли
    temp_subset['IDENTIFYCODE'] = temp_subset['IDENTIFYCODE'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    temp_subset['Source_Bank'] = bank_name
    df_p_list.append(temp_subset)

# Об'єднуємо всі банки
df_all_potential = pd.concat(df_p_list, ignore_index=True)

# ФІКС 3: Обробка перетинів. Якщо клієнт є у 2+ банках, з'єднуємо їх через плюсик
df_p = df_all_potential.groupby('IDENTIFYCODE')['Source_Bank'].apply(
    lambda x: ' + '.join(sorted(x.unique()))
).reset_index()


# ==========================================
# 2. Формуємо підмножину залучених клієнтів (A)
# ==========================================
# Читаємо конкретно перший лист, щоб не чіпати твій Validation_Report
df_a = pd.read_excel(zkp_file, sheet_name=0) 

df_a.rename(columns={'CONTRAGENTID': 'IDENTIFYCODE'}, inplace=True)
# Так само жорстко стандартизуємо ID
df_a['IDENTIFYCODE'] = df_a['IDENTIFYCODE'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

# Очищуємо доходи
income_cols = [col for col in df_a.columns if col.startswith('INCOME_')]
for col in income_cols:
    # Прибираємо дефіси і конвертуємо в числа
    df_a[col] = pd.to_numeric(df_a[col].astype(str).str.replace('-', ''), errors='coerce').fillna(0)


# ==========================================
# 3. СТВОРЕННЯ ТАБЛИЦІ ВИМІРУ (Dim_Clients)
# ==========================================
df_clients = pd.merge(
    df_p, 
    df_a[['IDENTIFYCODE', 'pilot_month']].drop_duplicates(), 
    on='IDENTIFYCODE', 
    how='outer'
)

df_clients['Source_Bank'] = df_clients['Source_Bank'].fillna('Unknown_Source')
df_clients['Is_Acquired'] = df_clients['pilot_month'].notna().astype(int)
df_clients['Status'] = np.where(df_clients['Is_Acquired'] == 1, 'Acquired', 'Potential Only')


# ==========================================
# 4. СТВОРЕННЯ ТАБЛИЦІ ФАКТІВ (Fact_Income)
# ==========================================
df_facts = pd.melt(
    df_a,
    id_vars=['IDENTIFYCODE'],
    value_vars=income_cols,
    var_name='Income_Month_Raw',
    value_name='Income_Value'
)

# Залишаємо лише факти прибутку
df_facts = df_facts[df_facts['Income_Value'] > 0].copy()

# Створюємо чіткі точкові дати замість інтервалів (як ти просив раніше)
df_facts['Date'] = df_facts['Income_Month_Raw'].str.replace('INCOME_', '') + '_01'
df_facts['Date'] = pd.to_datetime(df_facts['Date'], format='%Y_%m_%d').dt.date

df_facts = df_facts[['IDENTIFYCODE', 'Date', 'Income_Value']]


# ==========================================
# 5. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# ==========================================
df_clients.to_csv('Dim_Clients.csv', index=False)
df_facts.to_csv('Fact_Income.csv', index=False)

print(f"Успішно! Згенеровано таблиці:")
print(f"Dim_Clients.csv: {len(df_clients)} унікальних клієнтів")
print(f"Fact_Income.csv: {len(df_facts)} записів про дохід")