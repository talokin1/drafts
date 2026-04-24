import pandas as pd
import numpy as np

# --- 1. ПІДГОТОВКА КЛІЄНТСЬКИХ ДАНИХ (Об'єднання) ---

# Завантажуємо обидва файли (шляхи заміни на свої)
df1 = pd.read_excel(r"C:\Projects\(DS-702) Clients with payroll taxes\data\1 перелік клієнтів які.xlsx")
df2 = pd.read_excel(r"C:\Projects\(DS-702) Clients with payroll taxes\data\2 перелік клієнтів.xlsx")

# Перейменовуємо колонки та беремо лише потрібні
rename_dict = {"Номер клієнта": "CONTRAGENTID", "Код ЄДРПОУ клієнта": "IDENTIFYCODE"}
clients_1 = df1.rename(columns=rename_dict)[["IDENTIFYCODE", "CONTRAGENTID"]]
clients_2 = df2.rename(columns=rename_dict)[["IDENTIFYCODE", "CONTRAGENTID"]]

# Об'єднуємо в один датафрейм та видаляємо дублікати (якщо клієнт є в обох списках)
# Приводимо до рядкового типу (str), щоб уникнути помилок співставлення
all_clients = pd.concat([clients_1, clients_2]).drop_duplicates(subset=["IDENTIFYCODE"])
all_clients["IDENTIFYCODE"] = all_clients["IDENTIFYCODE"].astype(str)

# Множина всіх унікальних ID клієнтів
clients_ids = set(all_clients["IDENTIFYCODE"])


# --- 2. ФІЛЬТРАЦІЯ ТРАНЗАКЦІЙ ---

# Завантажуємо транзакції
data_trx = pd.read_parquet(r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2...")

# Фільтруємо за банком та приводимо ID до str
on_mfo_8 = data_trx[data_trx["BANKBID"].astype(str).str.startswith("8")]
on_mfo_8["CONTRAGENTBIDENTIFYCODE"] = on_mfo_8["CONTRAGENTBIDENTIFYCODE"].astype(str)

# Залишаємо лише транзакції НАШИХ клієнтів
client_trx = on_mfo_8[on_mfo_8["CONTRAGENTBIDENTIFYCODE"].isin(clients_ids)].copy()


# --- 3. КЛАСИФІКАЦІЯ ПОДАТКІВ (Векторизована функція) ---

purpose = client_trx['PLATPURPOSE'].astype(str).str.lower()

# Маски для різних типів податків
cond_pdfo = purpose.str.contains(r'пдфо|п\.д\.ф\.о\.', regex=True, na=False)
cond_esv = purpose.str.contains(r'єсв|єдин.*соц.*внес', regex=True, na=False)
cond_vz = purpose.str.contains(r'військ.*збір|\bвз\b|\bв\.з\.', regex=True, na=False)

# Список умов та відповідних їм міток
conditions = [cond_pdfo, cond_esv, cond_vz]
choices = ['ПДФО', 'ЄСВ', 'Військовий збір']

# Створюємо нову колонку TAX_TYPE
# Якщо жодна умова не виконується, ставимо 'Не податок'
client_trx['TAX_TYPE'] = np.select(conditions, choices, default='Не податок')


# --- 4. ЗНАХОДЖЕННЯ КЛІЄНТІВ БЕЗ ПОДАТКІВ ---

# Виділяємо транзакції, де ТИП податку не дорівнює 'Не податок'
tax_transactions = client_trx[client_trx['TAX_TYPE'] != 'Не податок']

# Отримуємо множину ID клієнтів, які ПЛАТИЛИ податки
tax_clients_ids = set(tax_transactions['CONTRAGENTBIDENTIFYCODE'])

# Знаходимо ID клієнтів, які Є в нашому початковому списку, але НЕ ПЛАТИЛИ податки
# Це також автоматично включить тих клієнтів, яких ВЗАГАЛІ немає в data_trx
clients_wo_taxes_ids = clients_ids - tax_clients_ids

# Відфільтровуємо початковий об'єднаний датафрейм клієнтів
clients_wo_taxes_df = all_clients[all_clients["IDENTIFYCODE"].isin(clients_wo_taxes_ids)].copy()

# Результат
display(clients_wo_taxes_df)
# Перегляд класифікованих транзакцій (для перевірки)
# display(client_trx[client_trx['TAX_TYPE'] != 'Не податок'].head())