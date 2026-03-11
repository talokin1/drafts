# ==========================================
# КРОК 2: Формування таблиці фактів (Fact_Income)
# ==========================================

# СТРОГИЙ ФІЛЬТР: Беремо тільки ті стовпці, назва яких починається з '202' (наприклад, '2025-06-30')
date_cols = [col for col in df_income.columns if str(col).startswith('202')]

# Трансформація матриці у векторний формат (Unpivot)
df_fact_income = pd.melt(
    df_income,
    id_vars=['CONTRAGENTID'],
    value_vars=date_cols,
    var_name='Income_Date',
    value_name='Income_Value'
)

# Жорстка типізація: перетворюємо значення на числа. Якщо попадеться текст - він стане NaN
df_fact_income['Income_Value'] = pd.to_numeric(df_fact_income['Income_Value'], errors='coerce')

# Відкидаємо порожні (NaN) та нульові значення
df_fact_income = df_fact_income.dropna(subset=['Income_Value'])
df_fact_income = df_fact_income[df_fact_income['Income_Value'] > 0]

# Приводимо дати до правильного формату
df_fact_income['Income_Date'] = pd.to_datetime(df_fact_income['Income_Date']).dt.date