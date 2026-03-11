import pandas as pd
import numpy as np

# 1. Завантаження множин
# Вкажи свої реальні шляхи до файлів, які ти згенерував
df_potential = pd.read_csv(r"C:\Projects\Alina Kinash\data_for_dashboard\potential_clients.csv")
df_engaged = pd.read_csv(r"C:\Projects\Alina Kinash\data_for_dashboard\engaged_clients_by_banks.csv") # тут був без розширення, додав .csv про всяк випадок
df_income = pd.read_csv(r"C:\Projects\Alina Kinash\data_for_dashboard\engaged_with_income.csv")

# Приведення ідентифікаторів до спільного текстового формату (щоб уникнути помилок типу 123.0 != "123")
df_potential['IDENTIFYCODE'] = df_potential['IDENTIFYCODE'].astype(str).str.replace(r'\.0$', '', regex=True)
df_engaged['IDENTIFYCODE'] = df_engaged['IDENTIFYCODE'].astype(str).str.replace(r'\.0$', '', regex=True)
df_engaged['CONTRAGENTID'] = df_engaged['CONTRAGENTID'].astype(str).str.replace(r'\.0$', '', regex=True)
df_income['CONTRAGENTID'] = df_income['CONTRAGENTID'].astype(str).str.replace(r'\.0$', '', regex=True)

# ==========================================
# КРОК 1: Формування таблиці вимірів (Dim_Clients)
# ==========================================

# Робимо Left Join: до всіх потенційних клієнтів підтягуємо дані про їхнє залучення
df_dim_clients = pd.merge(
    df_potential, 
    df_engaged[['IDENTIFYCODE', 'CONTRAGENTID', 'REGISTERDATE']], 
    on='IDENTIFYCODE', 
    how='left'
)

# Витягуємо статичну метрику карток з таблиці доходів
df_cards = df_income[['CONTRAGENTID', 'ZKP_NB_CARDS_2026_02']].copy()
df_cards.rename(columns={'ZKP_NB_CARDS_2026_02': 'Cards_Count'}, inplace=True)

# Підтягуємо кількість карток до клієнтів
df_dim_clients = pd.merge(df_dim_clients, df_cards, on='CONTRAGENTID', how='left')

# Формуємо статус (1 - Залучений, 0 - Тільки потенційний)
df_dim_clients['Is_Engaged'] = df_dim_clients['CONTRAGENTID'].notna().astype(int)
df_dim_clients['Status'] = np.where(df_dim_clients['Is_Engaged'] == 1, 'Engaged', 'Potential')

# ==========================================
# КРОК 2: Формування таблиці фактів (Fact_Income)
# ==========================================

# Визначаємо стовпці з датами (всі, окрім ID та карток)
date_cols = [col for col in df_income.columns if col not in ['CONTRAGENTID', 'ZKP_NB_CARDS_2026_02']]

# Трансформація матриці у векторний формат (Unpivot)
df_fact_income = pd.melt(
    df_income,
    id_vars=['CONTRAGENTID'],
    value_vars=date_cols,
    var_name='Income_Date',
    value_name='Income_Value'
)

# Очищуємо порожні значення (NaN), щоб не завантажувати пусті рядки в пам'ять
df_fact_income = df_fact_income.dropna(subset=['Income_Value'])
df_fact_income = df_fact_income[df_fact_income['Income_Value'] > 0]

# Приводимо дати до формату datetime
df_fact_income['Income_Date'] = pd.to_datetime(df_fact_income['Income_Date']).dt.date

# ==========================================
# ЗБЕРЕЖЕННЯ
# ==========================================
output_dir = r"C:\Projects\Alina Kinash\data_for_dashboard"
df_dim_clients.to_csv(f"{output_dir}\PBI_Dim_Clients.csv", index=False)
df_fact_income.to_csv(f"{output_dir}\PBI_Fact_Income.csv", index=False)

print("Дані для Power BI успішно підготовлені.")