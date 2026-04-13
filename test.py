import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def create_business_report(input_df, active_clients, output_filename='Business_Model_Report.xlsx'):
    df = input_df.copy()
    
    # 1. Data Cleaning
    df['CONTRAGENTID'] = df['CONTRAGENTID'].replace('Missing value', np.nan)
    df['MONTHLY_INCOME'] = pd.to_numeric(df['MONTHLY_INCOME'].replace('Missing value', np.nan), errors='coerce')
    df['POTENTIAL_INCOME'] = pd.to_numeric(df['POTENTIAL_INCOME'], errors='coerce')

    # Фільтруємо існуючих клієнтів (тих, що мають і реальний, і потенційний дохід) з переданого active_clients
    eval_df = active_clients.dropna(subset=['CONTRAGENTID', 'MONTHLY_INCOME', 'POTENTIAL_INCOME']).copy()

    # 2. MODEL ACCURACY SHEET (Error Analysis)
    accuracy_cols = ['IDENTIFYCODE', 'FIRM_NAME', 'FIRM_TYPE', 'MONTHLY_INCOME', 'POTENTIAL_INCOME']
    accuracy_df = eval_df[accuracy_cols].copy()
    
    # Рахуємо абсолютну помилку
    accuracy_df['ABS_ERROR'] = (accuracy_df['MONTHLY_INCOME'] - accuracy_df['POTENTIAL_INCOME']).abs()
    
    # Сортуємо за найбільшою помилкою (як на скріншоті)
    accuracy_df = accuracy_df.sort_values(by='ABS_ERROR', ascending=False).reset_index(drop=True)

    # Рахуємо метрики ТІЛЬКИ для ненульових клієнтів
    non_zero_df = accuracy_df[accuracy_df['MONTHLY_INCOME'] != 0]
    if len(non_zero_df) > 0:
        mae_val = mean_absolute_error(non_zero_df['MONTHLY_INCOME'], non_zero_df['POTENTIAL_INCOME'])
        count_val = len(non_zero_df)
    else:
        mae_val, count_val = 0, 0

    # 3. PIVOT TABLES (English headers)
    pivot_division = df.groupby(['DIVISION_CODE', 'DIVISION_NAME']).agg(
        Company_Count=('IDENTIFYCODE', 'count'),
        Average_Potential=('POTENTIAL_INCOME', 'mean'),
        Total_Potential=('POTENTIAL_INCOME', 'sum')
    ).sort_values(by='Average_Potential', ascending=False).reset_index()

    pivot_firm_type = df.groupby('FIRM_TYPE').agg(
        Company_Count=('IDENTIFYCODE', 'count'),
        Average_Potential=('POTENTIAL_INCOME', 'mean'),
        Total_Potential=('POTENTIAL_INCOME', 'sum')
    ).sort_values(by='Average_Potential', ascending=False).reset_index()

    pivot_our_clients = eval_df.groupby('FIRM_TYPE').agg(
        Our_Clients_Count=('IDENTIFYCODE', 'count'),
        Average_Real_Income=('MONTHLY_INCOME', 'mean'),
        Average_Potential=('POTENTIAL_INCOME', 'mean')
    ).sort_values(by='Our_Clients_Count', ascending=False).reset_index()

    # 4. EXCEL WRITING
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        money_fmt_int = workbook.add_format({'num_format': '#,##0', 'align': 'right'}) # Без копійок для похибок
        money_fmt_dec = workbook.add_format({'num_format': '#,##0.00', 'align': 'right'}) # З копійками для аналітики
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white', 'border': 1})
        title_fmt = workbook.add_format({'bold': True, 'font_size': 14})
        metrics_header_fmt = workbook.add_format({'bold': True, 'bg_color': '#E26B0A', 'font_color': 'white', 'border': 1})

        # --- Sheet 1: Model Accuracy ---
        accuracy_df.to_excel(writer, sheet_name='Model Accuracy', index=False)
        worksheet_accuracy = writer.sheets['Model Accuracy']
        
        worksheet_accuracy.autofilter(0, 0, len(accuracy_df), len(accuracy_df.columns) - 1)
        worksheet_accuracy.freeze_panes(1, 0)
        
        # Форматування основної таблиці (Колонки A-F)
        worksheet_accuracy.set_column('A:A', 15)
        worksheet_accuracy.set_column('B:B', 40)
        worksheet_accuracy.set_column('C:C', 15)
        worksheet_accuracy.set_column('D:F', 20, money_fmt_int)
        
        for col_num, value in enumerate(accuracy_df.columns.values):
            worksheet_accuracy.write(0, col_num, value, header_fmt)

        # ДОДАВАННЯ МЕТРИК ЗБОКУ (Колонки H та I)
        # 7 = H, 8 = I у нумерації xlsxwriter
        worksheet_accuracy.write_string(0, 7, "Metric", metrics_header_fmt)
        worksheet_accuracy.write_string(0, 8, "Value", metrics_header_fmt)
        
        worksheet_accuracy.write_string(1, 7, "Valid Clients Count (Income != 0)")
        worksheet_accuracy.write_number(1, 8, count_val)
        
        worksheet_accuracy.write_string(2, 7, "MAE (Non-Zero Clients)")
        worksheet_accuracy.write_number(2, 8, mae_val, money_fmt_int)
        
        worksheet_accuracy.set_column(7, 7, 30) # Ширина для назви метрики (Колонка H)
        worksheet_accuracy.set_column(8, 8, 20) # Ширина для значення (Колонка I)

        # --- Sheet 2: Business Analytics ---
        worksheet_pivot = workbook.add_worksheet('Business Analytics')
        writer.sheets['Business Analytics'] = worksheet_pivot
        
        row_idx = 0
        worksheet_pivot.write_string(row_idx, 0, "1. Average and Total Profitability by Segment (FIRM_TYPE)", title_fmt)
        row_idx += 1
        pivot_firm_type.to_excel(writer, sheet_name='Business Analytics', startrow=row_idx, index=False)
        for col_num, value in enumerate(pivot_firm_type.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)
        row_idx += len(pivot_firm_type) + 3
        
        worksheet_pivot.write_string(row_idx, 0, "2. Distribution of EXISTING Bank Clients", title_fmt)
        row_idx += 1
        pivot_our_clients.to_excel(writer, sheet_name='Business Analytics', startrow=row_idx, index=False)
        for col_num, value in enumerate(pivot_our_clients.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)
        row_idx += len(pivot_our_clients) + 3

        worksheet_pivot.write_string(row_idx, 0, "3. Industry Ranking (DIVISION)", title_fmt)
        row_idx += 1
        pivot_division.to_excel(writer, sheet_name='Business Analytics', startrow=row_idx, index=False)
        for col_num, value in enumerate(pivot_division.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)

        worksheet_pivot.set_column('A:B', 35)
        worksheet_pivot.set_column('C:E', 20, money_fmt_dec)

        # --- Sheet 3: Raw Data ---
        df.to_excel(writer, sheet_name='Raw Data', index=False)
        worksheet_raw = writer.sheets['Raw Data']
        worksheet_raw.autofilter(0, 0, len(df), len(df.columns) - 1)
        worksheet_raw.freeze_panes(1, 0)
        
        for col_num, col_name in enumerate(df.columns):
            worksheet_raw.set_column(col_num, col_num, 15) 
            if 'POTENTIAL' in col_name or 'INCOME' in col_name:
                worksheet_raw.set_column(col_num, col_num, 20, money_fmt_dec)

    print(f"Report successfully generated: {output_filename}")