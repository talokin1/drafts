# 4. ЗВЕДЕНА ТАБЛИЦЯ (ANALYTICS)
# Прибираємо ['Difs'] перед .agg, щоб мати доступ до всіх колонок
summary_table = report_df.groupby('Income_Range', observed=False).agg(
    Count=('Difs', 'count'),            # Рахуємо кількість (можна по будь-якій колонці)
    Mean_Error=('Difs', 'mean'),        # Середня помилка
    Median_Error=('Difs', 'median'),    # Медіанна помилка
    Avg_Predicted=('Predicted', 'mean') # Тепер це спрацює, бо 'Predicted' доступна
).reset_index()

# Додаткове округлення для краси
summary_table = summary_table.round(2)