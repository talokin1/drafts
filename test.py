numeric_cols_to_clean = ['price_usd', 'min_month_leasing_pay', 'mileage', 'doors_count']
for col in numeric_cols_to_clean:
    if col in df_inf.columns:
        # Переводимо в рядок, видаляємо ВСІ пробіли (щоб '53 811' стало '53811')
        df_inf[col] = df_inf[col].astype(str).str.replace(' ', '', regex=False)
        # pd.to_numeric з errors='coerce' автоматично перетворить текст (наприклад, 'Missingvalue') на NaN
        df_inf[col] = pd.to_numeric(df_inf[col], errors='coerce')