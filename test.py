import pandas as pd
import numpy as np

def generate_business_excel(df_all, df_active, output_filename="Potential_Income_Report.xlsx"):
    # Копіюємо дані, щоб не змінювати оригінали
    df_all = df_all.copy()
    df_active = df_active.copy()
    
    # 1. Зведені таблиці (на основі всієї бази)
    pivot_firm_type = pd.pivot_table(df_all, 
                                     index='FIRM_TYPE', 
                                     values=['IDENTIFYCODE', 'POTENTIAL_INCOME'],
                                     aggfunc={'IDENTIFYCODE': 'count', 
                                              'POTENTIAL_INCOME': 'sum'}).rename(columns={'IDENTIFYCODE': 'Кількість компаній'})
    
    pivot_kved = pd.pivot_table(df_all, 
                                index='KVED_DESCR', 
                                values=['POTENTIAL_INCOME'],
                                aggfunc=['count', 'sum'])
    pivot_kved.columns = ['Кількість компаній', 'Сумарний потенціал']
    pivot_kved = pivot_kved.sort_values(by='Сумарний потенціал', ascending=False).head(15)

    # 2. Розрахунок точності (ТІЛЬКИ на основі active_clients)
    df_active['ABS_ERROR'] = abs(df_active['MONTHLY_INCOME'] - df_active['POTENTIAL_INCOME'])
    
    # Уникаємо ділення на нуль
    df_active['ERROR_PCT'] = np.where(df_active['MONTHLY_INCOME'] > 0, 
                                      df_active['ABS_ERROR'] / df_active['MONTHLY_INCOME'], np.nan)
    
    # Сортуємо від найбільшої відсоткової помилки до найменшої (або за ABS_ERROR)
    df_active = df_active.sort_values(by='ABS_ERROR', ascending=False)
    
    # Рахуємо портфельні метрики для бізнесу
    total_actual_income = df_active['MONTHLY_INCOME'].sum()
    total_predicted_income = df_active['POTENTIAL_INCOME'].sum()
    total_abs_error = df_active['ABS_ERROR'].sum()
    wmape = total_abs_error / total_actual_income if total_actual_income > 0 else 0

    # 3. Запис у Excel
    writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
    workbook = writer.book
    
    # Формати
    money_fmt = workbook.add_format({'num_format': '#,##0.00 ₴'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#1F497D', 'font_color': 'white', 'border': 1})
    metric_title_fmt = workbook.add_format({'bold': True, 'font_size': 12, 'align': 'right'})
    metric_val_fmt = workbook.add_format({'bold': True, 'font_size': 12, 'num_format': '#,##0.00 ₴', 'font_color': '#1F497D'})
    metric_pct_fmt = workbook.add_format({'bold': True, 'font_size': 12, 'num_format': '0.00%', 'font_color': '#C00000'})
    
    # --- ЛИСТ 1: Сирі дані ---
    df_all.to_excel(writer, sheet_name='Raw_Data', index=False)
    ws1 = writer.sheets['Raw_Data']
    ws1.autofilter(0, 0, len(df_all), len(df_all.columns) - 1)
    ws1.freeze_panes(1, 0)
    
    ws1.set_column('A:B', 15)
    ws1.set_column('C:G', 18, money_fmt)
    ws1.set_column('H:H', 30)
    ws1.set_column('P:P', 18, money_fmt)
    for col_num, value in enumerate(df_all.columns.values):
        ws1.write(0, col_num, value, header_fmt)

    # --- ЛИСТ 2: Зведена аналітика ---
    pivot_firm_type.to_excel(writer, sheet_name='Summary_Dashboard', startrow=1, startcol=1)
    pivot_kved.to_excel(writer, sheet_name='Summary_Dashboard', startrow=len(pivot_firm_type) + 5, startcol=1)
    ws2 = writer.sheets['Summary_Dashboard']
    ws2.write(0, 1, "Аналітика за типом компанії", workbook.add_format({'bold': True, 'font_size': 14}))
    ws2.write(len(pivot_firm_type) + 4, 1, "Топ-15 галузей (КВЕД) за потенціалом", workbook.add_format({'bold': True, 'font_size': 14}))
    ws2.set_column('B:B', 35)
    ws2.set_column('C:E', 20, money_fmt)

    # --- ЛИСТ 3: Точність моделі (на active_clients) ---
    # Відступаємо 5 рядків зверху для виведення головних метрик
    start_row = 6 
    
    cols_to_export = ['IDENTIFYCODE', 'FIRM_NAME', 'FIRM_TYPE', 'MONTHLY_INCOME', 'POTENTIAL_INCOME', 'ABS_ERROR', 'ERROR_PCT']
    df_active[cols_to_export].to_excel(writer, sheet_name='Model_Accuracy', index=False, startrow=start_row)
    ws3 = writer.sheets['Model_Accuracy']
    
    # Виводимо портфельні метрики (Dashboard зверху)
    ws3.write(1, 1, "Сумарний фактичний дохід портфеля:", metric_title_fmt)
    ws3.write(1, 2, total_actual_income, metric_val_fmt)
    
    ws3.write(2, 1, "Сумарний передбачений дохід:", metric_title_fmt)
    ws3.write(2, 2, total_predicted_income, metric_val_fmt)
    
    ws3.write(3, 1, "Сумарна абсолютна помилка:", metric_title_fmt)
    ws3.write(3, 2, total_abs_error, metric_val_fmt)
    
    ws3.write(4, 1, "WMAPE (Зважена похибка на портфель):", metric_title_fmt)
    ws3.write(4, 2, wmape, metric_pct_fmt)

    # Форматування таблиці точності
    ws3.autofilter(start_row, 0, len(df_active) + start_row, len(cols_to_export) - 1)
    ws3.freeze_panes(start_row + 1, 0)
    
    ws3.set_column('A:A', 15)
    ws3.set_column('B:B', 30)
    ws3.set_column('C:C', 15)
    ws3.set_column('D:F', 20, money_fmt)
    ws3.set_column('G:G', 15, pct_fmt)
    
    # Умовне форматування для відсотків помилки
    ws3.conditional_format(start_row + 1, 6, len(df_active) + start_row, 6, {
        'type': '3_color_scale',
        'min_color': '#63BE7B',
        'mid_color': '#FFEB84',
        'max_color': '#F8696B',
        'min_type': 'num', 'min_value': 0.0,
        'mid_type': 'num', 'mid_value': 0.2, 
        'max_type': 'num', 'max_value': 0.5  
    })
    
    # Заголовки таблиці
    for col_num, value in enumerate(cols_to_export):
        ws3.write(start_row, col_num, value, header_fmt)

    writer.close()
    print(f"Файл {output_filename} успішно згенеровано!")

# Виклик функції:
# generate_business_excel(df_all=your_full_dataframe, df_active=active_clients)