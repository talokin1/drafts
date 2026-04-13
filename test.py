import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def create_business_report(input_df, output_filename='Business_Model_Report.xlsx'):
    df = input_df.copy()
    
    # 1. Очистка даних
    df['CONTRAGENTID'] = df['CONTRAGENTID'].replace('Missing value', np.nan)
    df['MONTHLY_INCOME'] = pd.to_numeric(df['MONTHLY_INCOME'].replace('Missing value', np.nan), errors='coerce')
    df['POTENTIAL_INCOME'] = pd.to_numeric(df['POTENTIAL_INCOME'], errors='coerce')

    # Фільтруємо існуючих клієнтів
    existing_clients_df = df.dropna(subset=['CONTRAGENTID']).copy()

    # 2. МЕТРИКИ ТОЧНОСТІ (тільки для тих, де є і прогноз, і реальність)
    eval_df = existing_clients_df.dropna(subset=['MONTHLY_INCOME', 'POTENTIAL_INCOME'])
    if len(eval_df) > 0:
        y_true = eval_df['MONTHLY_INCOME']
        y_pred = eval_df['POTENTIAL_INCOME']
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mask_non_zero = y_true != 0
        mape = np.mean(np.abs((y_true[mask_non_zero] - y_pred[mask_non_zero]) / y_true[mask_non_zero])) * 100
    else:
        mae, r2, mape = 0, 0, 0

    metrics_df = pd.DataFrame({
        'Метрика': ['Кількість клієнтів для тесту', 'MAE (Середня помилка у валюті)', 'MAPE (Помилка у %)', 'R^2'],
        'Значення': [len(eval_df), mae, mape, r2]
    })

    # 3. ЗВЕДЕНІ ТАБЛИЦІ ЗА ВИМОГАМИ КЕРІВНИЦТВА

    # Вимога 1: Рейтинг індустрій (DIVISION) за СЕРЕДНЬОЮ прибутковістю
    pivot_division = df.groupby(['DIVISION_CODE', 'DIVISION_NAME']).agg(
        Кількість_компаній=('IDENTIFYCODE', 'count'),
        Середній_Потенціал=('POTENTIAL_INCOME', 'mean'),
        Сумарний_Потенціал=('POTENTIAL_INCOME', 'sum')
    ).sort_values(by='Середній_Потенціал', ascending=False).reset_index()

    # Вимога 2: Аналіз за розміром бізнесу (FIRM_TYPE) - середнє та Total
    pivot_firm_type = df.groupby('FIRM_TYPE').agg(
        Кількість_компаній=('IDENTIFYCODE', 'count'),
        Середній_Потенціал=('POTENTIAL_INCOME', 'mean'),
        Сумарний_Потенціал=('POTENTIAL_INCOME', 'sum')
    ).sort_values(by='Середній_Потенціал', ascending=False).reset_index()

    # Вимога 3: Розподіл НАШИХ клієнтів (за FIRM_TYPE для наочності)
    pivot_our_clients = existing_clients_df.groupby('FIRM_TYPE').agg(
        Кількість_наших_клієнтів=('IDENTIFYCODE', 'count'),
        Середній_Реальний_Дохід=('MONTHLY_INCOME', 'mean'),
        Середній_Потенціал=('POTENTIAL_INCOME', 'mean')
    ).sort_values(by='Кількість_наших_клієнтів', ascending=False).reset_index()


    # 4. ЗАПИС У EXCEL
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        money_fmt = workbook.add_format({'num_format': '#,##0.00', 'align': 'right'})
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white', 'border': 1})
        title_fmt = workbook.add_format({'bold': True, 'font_size': 14})

        # --- Лист 1: Точність ---
        metrics_df.to_excel(writer, sheet_name='Точність Моделі', index=False)
        
        # --- Лист 2: Аналітика (Зведені дані) ---
        worksheet_pivot = workbook.add_worksheet('Бізнес Аналітика')
        writer.sheets['Бізнес Аналітика'] = worksheet_pivot
        
        row_idx = 0
        
        # Вимога 2: Сегменти
        worksheet_pivot.write_string(row_idx, 0, "1. Середня та загальна прибутковість за Сегментами (FIRM_TYPE)", title_fmt)
        row_idx += 1
        pivot_firm_type.to_excel(writer, sheet_name='Бізнес Аналітика', startrow=row_idx, index=False)
        # Застосування стилів заголовків
        for col_num, value in enumerate(pivot_firm_type.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)
        row_idx += len(pivot_firm_type) + 3
        
        # Вимога 3: Наші клієнти
        worksheet_pivot.write_string(row_idx, 0, "2. Розподіл ІСНУЮЧИХ клієнтів банку", title_fmt)
        row_idx += 1
        pivot_our_clients.to_excel(writer, sheet_name='Бізнес Аналітика', startrow=row_idx, index=False)
        for col_num, value in enumerate(pivot_our_clients.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)
        row_idx += len(pivot_our_clients) + 3

        # Вимога 1: Індустрії (Рейтинг)
        worksheet_pivot.write_string(row_idx, 0, "3. Рейтинг Індустрій (DIVISION) за середньою прибутковістю", title_fmt)
        row_idx += 1
        pivot_division.to_excel(writer, sheet_name='Бізнес Аналітика', startrow=row_idx, index=False)
        for col_num, value in enumerate(pivot_division.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)

        # Форматування ширини колонок на листі аналітики
        worksheet_pivot.set_column('A:B', 35)
        worksheet_pivot.set_column('C:E', 20, money_fmt)

        # --- Лист 3: Сирі дані ---
        df.to_excel(writer, sheet_name='База (Сирі Дані)', index=False)
        worksheet_raw = writer.sheets['База (Сирі Дані)']
        worksheet_raw.autofilter(0, 0, len(df), len(df.columns) - 1)
        worksheet_raw.freeze_panes(1, 0)
        
        for col_num, col_name in enumerate(df.columns):
            worksheet_raw.set_column(col_num, col_num, 15) 
            if 'POTENTIAL' in col_name or 'INCOME' in col_name:
                worksheet_raw.set_column(col_num, col_num, 20, money_fmt)

    print(f"Звіт успішно згенеровано: {output_filename}")

# create_business_report(df, "Corp_Income_Model_Report.xlsx")