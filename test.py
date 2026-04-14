# 1.1 Примусове приведення колонок до числового типу перед агрегацією
# Параметр errors='coerce' перетворить будь-який нечисловий текст (як-от 'Missing value') на NaN

numeric_cols_for_max = ['doors_count', 'has_report', 'is_abroad']

for col in numeric_cols_for_max:
    # Замінюємо булеві True/False на 1/0, якщо вони є. Текст стає NaN.
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Правила агрегації (залишаються без змін)
agg_funcs = {
    'is_hnwi': 'max',
    'price_usd': ['count', 'mean', 'max', 'sum'],
    'min_month_leasing_pay': ['mean', 'max'],
    'leasing_to_price_ratio': ['mean', 'max'],
    'mileage': ['mean', 'min'],
    'is_luxury_car': 'max', 
    'doors_count': 'max',  # Тепер тут чисті float, NaN будуть проігноровані
    'has_report': 'max',   # Для 1/0 max працює як логічне АБО (any)
    'is_abroad': 'max',
    'category_name': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'Missing',
    'exchange_type': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'Missing'
}

# 3. Групуємо
df_client = df.groupby('MOBILEPHONE').agg(agg_funcs)