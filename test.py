import pandas as pd
import os
import glob
import numpy as np

# Шляхи до даних (заміни на свої за потреби)
folder_path = 'potential_clients/'
zkp_file = 'ZKP_PROJECT_BY_MARCH.xlsx'

print("Починаю обробку даних...")

# ==========================================
# 1. Формуємо підмножину потенційних клієнтів (P)
# ==========================================
all_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
df_p_list = []

for file in all_files:
    df_temp = pd.read_excel(file)
    bank_name = os.path.basename(file).replace('_clients.xlsx', '').replace('.xlsx', '')
    df_temp['Source_Bank'] = bank_name
    df_p_list.append(df_temp)

df_p = pd.concat(df_p_list, ignore_index=True)
# Витягуємо лише унікальних клієнтів та їхній банк
df_p = df_p[['IDENTIFYCODE', 'Source_Bank']].drop_duplicates()


# ==========================================
# 2. Формуємо підмножину залучених клієнтів (A)
# ==========================================
df_a = pd.read_excel(zkp_file)
df_a.rename(columns={'CONTRAGENTID': 'IDENTIFYCODE'}, inplace=True)

# Очищуємо доходи: перетворюємо дефіси на 0 та конвертуємо у числа
income_cols = [col for col in df_a.columns if col.startswith('INCOME_')]
for col in income_cols:
    df_a[col] = pd.to_numeric(df_a[col].replace('-', np.nan), errors='coerce').fillna(0)


# ==========================================
# 3. СТВОРЕННЯ ТАБЛИЦІ ВИМІРУ (Dim_Clients)
# ==========================================
# Об'єднуємо обидві множини (Outer Join), щоб не втратити жодного клієнта
df_clients = pd.merge(
    df_p, 
    df_a[['IDENTIFYCODE', 'pilot_month']].drop_duplicates(), 
    on='IDENTIFYCODE', 
    how='outer'
)

# Заповнюємо порожнечі, якщо клієнт є в ZKP, але немає в папці
df_clients['Source_Bank'] = df_clients['Source_Bank'].fillna('Unknown_Source')

# Додаємо чіткі індикатори статусу для побудови Воронки (Funnel)
df_clients['Is_Acquired'] = df_clients['pilot_month'].notna().astype(int)
df_clients['Status'] = np.where(df_clients['Is_Acquired'] == 1, 'Acquired', 'Potential Only')


# ==========================================
# 4. СТВОРЕННЯ ТАБЛИЦІ ФАКТІВ (Fact_Income)
# ==========================================
# Перетворюємо матрицю доходів у векторний формат
df_facts = pd.melt(
    df_a,
    id_vars=['IDENTIFYCODE'],
    value_vars=income_cols,
    var_name='Income_Month_Raw',
    value_name='Income_Value'
)

# Відкидаємо нульові значення (зберігаємо лише факти отримання прибутку)
df_facts = df_facts[df_facts['Income_Value'] > 0].copy()

# Формуємо чітку дату (перше число місяця) замість текстового інтервалу
# 'INCOME_2025_06' -> '2025-06-01'
df_facts['Date'] = df_facts['Income_Month_Raw'].str.replace('INCOME_', '') + '_01'
df_facts['Date'] = pd.to_datetime(df_facts['Date'], format='%Y_%m_%d').dt.date

# Залишаємо лише необхідні координати
df_facts = df_facts[['IDENTIFYCODE', 'Date', 'Income_Value']]


# ==========================================
# 5. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# ==========================================
df_clients.to_csv('Dim_Clients.csv', index=False)
df_facts.to_csv('Fact_Income.csv', index=False)

print(f"Готово! Згенеровано 2 таблиці для Power BI:")
print(f"1) Dim_Clients.csv (Клієнтів: {len(df_clients)})")
print(f"2) Fact_Income.csv (Транзакцій: {len(df_facts)})")