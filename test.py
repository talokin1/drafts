import pandas as pd
import numpy as np

# --- 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ ---

# Завантажуємо вихідні файли, зберігаючи всі оригінальні колонки. 
# Одразу читаємо ID як рядки (str), щоб уникнути втрати нулів на початку
dtypes_dict = {"Номер клієнта": str, "Код ЄДРПОУ клієнта": str}
df1 = pd.read_excel(r"C:\Projects\(DS-702) Clients with payroll taxes\data\1 перелік клієнтів які імпортують.xlsx", dtype=dtypes_dict)
df2 = pd.read_excel(r"C:\Projects\(DS-702) Clients with payroll taxes\data\2 перелік клієнтів .xlsx", dtype=dtypes_dict)

# Створюємо єдиний реєстр клієнтів для пошуку (зберігаємо всі колонки)
all_clients_full = pd.concat([df1, df2], ignore_index=True)
clients_ids = set(all_clients_full["Код ЄДРПОУ клієнта"].dropna())


# --- 2. ФІЛЬТРАЦІЯ ТРАНЗАКЦІЙ ---

data_trx = pd.read_parquet(r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2...")

# Фільтр по банку та приведення до str
on_mfo_8 = data_trx[data_trx["BANKBID"].astype(str).str.startswith("8")].copy()
on_mfo_8["CONTRAGENTAIDENTIFYCODE"] = on_mfo_8["CONTRAGENTAIDENTIFYCODE"].astype(str)
on_mfo_8["CONTRAGENTBIDENTIFYCODE"] = on_mfo_8["CONTRAGENTBIDENTIFYCODE"].astype(str)

# Залишаємо транзакції, де наш клієнт є АБО відправником, АБО отримувачем
client_trx = on_mfo_8[
    on_mfo_8["CONTRAGENTAIDENTIFYCODE"].isin(clients_ids) | 
    on_mfo_8["CONTRAGENTBIDENTIFYCODE"].isin(clients_ids)
].copy()


# --- 3. КЛАСИФІКАЦІЯ ПОДАТКІВ ---

purpose = client_trx['PLATPURPOSE'].astype(str).str.lower()

cond_pdfo = purpose.str.contains(r'пдфо|п\.д\.ф\.о\.', regex=True, na=False)
cond_esv = purpose.str.contains(r'єсв|єдин.*соц.*внес', regex=True, na=False)
cond_vz = purpose.str.contains(r'військ.*збір|\bвз\b|\bв\.з\.', regex=True, na=False)

conditions = [cond_pdfo, cond_esv, cond_vz]
choices = ['ПДФО', 'ЄСВ', 'Військовий збір']

client_trx['TAX_TYPE'] = np.select(conditions, choices, default='Не податок')


# --- 4. АГРЕГАЦІЯ ПОДАТКІВ ПО КОЖНОМУ КЛІЄНТУ ---

# Відкидаємо транзакції, які не є податками
tax_transactions = client_trx[client_trx['TAX_TYPE'] != 'Не податок']

# Оскільки клієнт може бути як відправником, так і отримувачем, 
# збираємо всі комбінації [ID клієнта - Тип податку]
tax_a = tax_transactions[['CONTRAGENTAIDENTIFYCODE', 'TAX_TYPE']].rename(columns={'CONTRAGENTAIDENTIFYCODE': 'ID'})
tax_b = tax_transactions[['CONTRAGENTBIDENTIFYCODE', 'TAX_TYPE']].rename(columns={'CONTRAGENTBIDENTIFYCODE': 'ID'})

# Об'єднуємо, фільтруємо тільки НАШИХ клієнтів і видаляємо пусті значення
all_tax_records = pd.concat([tax_a, tax_b]).dropna()
our_client_taxes = all_tax_records[all_tax_records['ID'].isin(clients_ids)]

# АГРЕГАЦІЯ: Групуємо по ID і збираємо унікальні податки в один рядок через кому
# Наприклад: "ПДФО, ЄСВ"
client_tax_mapping = our_client_taxes.groupby('ID')['TAX_TYPE'].apply(
    lambda x: ', '.join(sorted(set(x)))
).reset_index()


# --- 5. ЗБАГАЧЕННЯ ВИХІДНИХ ФАЙЛІВ (df1 та df2) ---

# Робимо Left Join (аналог ВПР/VLOOKUP в Excel), щоб підтягнути колонку TAX_TYPE
df1 = df1.merge(client_tax_mapping, left_on="Код ЄДРПОУ клієнта", right_on="ID", how="left")
df2 = df2.merge(client_tax_mapping, left_on="Код ЄДРПОУ клієнта", right_on="ID", how="left")

# Заповнюємо порожні значення (клієнти без податків) та видаляємо технічну колонку ID
fill_value = 'Не сплачує податки в ОТП'
df1['TAX_TYPE'] = df1['TAX_TYPE'].fillna(fill_value)
df2['TAX_TYPE'] = df2['TAX_TYPE'].fillna(fill_value)

df1 = df1.drop(columns=['ID'], errors='ignore')
df2 = df2.drop(columns=['ID'], errors='ignore')


# --- 6. ВИДІЛЕННЯ КЛІЄНТІВ БЕЗ ПОДАТКІВ (Ті самі ~601 клієнт) ---

# Знаходимо ID клієнтів, які сплатили хоча б один податок
tax_clients_ids = set(client_tax_mapping['ID'])

# Відбираємо клієнтів, яких немає в множині платників податків
clients_wo_taxes_df = all_clients_full[~all_clients_full["Код ЄДРПОУ клієнта"].isin(tax_clients_ids)].copy()

# Видаляємо дублікати за ЄДРПОУ, щоб отримати чистий список унікальних клієнтів
clients_wo_taxes_unique = clients_wo_taxes_df.drop_duplicates(subset=["Код ЄДРПОУ клієнта"])

# Залишаємо тільки необхідні колонки для фінального звіту
columns_to_export = ['Номер клієнта', 'Код ЄДРПОУ клієнта', 'Назва клієнта', 'Сегмент клієнта']
final_export_df = clients_wo_taxes_unique[columns_to_export]


# --- 7. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ (Опціонально) ---
# df1.to_excel("df1_with_taxes.xlsx", index=False)
# df2.to_excel("df2_with_taxes.xlsx", index=False)
# final_export_df.to_excel("clients_without_taxes.xlsx", index=False)

# Перевірка результатів
display(final_export_df)
print(f"Кількість унікальних клієнтів без податків: {len(final_export_df)}")