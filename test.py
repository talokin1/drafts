import pandas as pd
import numpy as np

def generate_peter_matrix_report(df_active, output_filename="Peter_Matrix_Report.xlsx"):
    df = df_active.copy()
    
    # 1. Створення категорій (Бакетування)
    bins = [-np.inf, 300, 1000, 2500, 10000, np.inf]
    labels = ['total hopeless', 'bad quality', 'grey zone', 'green zone', 'fantastic']
    
    # Використовуємо pd.Categorical, щоб зафіксувати строгий порядок сортування в таблицях
    df['Fact_Peter_category'] = pd.cut(df['MONTHLY_INCOME'], bins=bins, labels=labels, right=False)
    df['Predicted_Peter_category'] = pd.cut(df['POTENTIAL_INCOME'], bins=bins, labels=labels, right=False)
    
    df['Fact_Peter_category'] = pd.Categorical(df['Fact_Peter_category'], categories=labels, ordered=True)
    df['Predicted_Peter_category'] = pd.Categorical(df['Predicted_Peter_category'], categories=labels, ordered=True)

    # 2. Розрахунок абсолютної матриці (Count)
    count_matrix = pd.crosstab(df['Fact_Peter_category'], 
                               df['Predicted_Peter_category'], 
                               margins=True, 
                               margins_name='Grand Total',
                               dropna=False) # dropna=False гарантує, що всі класи будуть відображені, навіть якщо там 0

    # 3. Розрахунок відносної матриці (Percentages)
    total_clients = count_matrix.loc['Grand Total', 'Grand Total']
    pct_matrix = count_matrix / total_clients

    # 4. Бізнес-логіка: Матриця кольорів (Рядки - Факт, Колонки - Предикт)
    # Згідно з вашим описом та скріншотом: 'G' - Green, 'Y' - Yellow, 'R' - Red
    color_logic = [
        ['G', 'G', 'R', 'R', 'R'],  # total hopeless
        ['G', 'G', 'Y', 'R', 'R'],  # bad quality
        ['Y', 'Y', 'G', 'G', 'G'],  # grey zone
        ['R', 'R', 'Y', 'G', 'G'],  # green zone
        ['R', 'R', 'R', 'G', 'G']   # fantastic
    ]
    
    # Підрахунок загальних відсотків по зонах
    zone_totals = {'G': 0.0, 'Y': 0.0, 'R': 0.0}
    for i in range(5):
        for j in range(5):
            zone = color_logic[i][j]
            val = pct_matrix.iloc[i, j]
            if pd.notna(val):
                zone_totals[zone] += val

    # 5. Запис у Excel
    writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
    workbook = writer.book
    worksheet = workbook.add_worksheet('Peter_Matrix')
    
    # Формати
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#00B0F0', 'font_color': 'white', 'border': 1, 'align': 'center'})
    row_label_fmt = workbook.add_format({'bold': True, 'bg_color': '#FFC000', 'border': 1})
    count_fmt = workbook.add_format({'border': 1, 'num_format': '0'})
    grand_total_fmt = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#E7E6E6'})
    
    # Формати для кольорових зон (відсотки)
    pct_fmt_base = {'border': 1, 'num_format': '0.0%'}
    fmt_green = workbook.add_format({**pct_fmt_base, 'bg_color': '#C4D79B'})  # Приглушений зелений
    fmt_yellow = workbook.add_format({**pct_fmt_base, 'bg_color': '#FFFF00'}) # Жовтий
    fmt_red = workbook.add_format({**pct_fmt_base, 'bg_color': '#FF0000', 'font_color': 'white'}) # Червоний
    
    fmt_map = {'G': fmt_green, 'Y': fmt_yellow, 'R': fmt_red}

    # --- ЗАПИС МАТРИЦІ КІЛЬКОСТЕЙ (Count) ---
    worksheet.write(1, 3, "Predictions", header_fmt)
    worksheet.merge_range('D2:H2', "Predictions", header_fmt)
    worksheet.write(2, 1, "Fact", row_label_fmt)
    worksheet.merge_range('B3:B7', "Fact", row_label_fmt)
    
    # Заголовки колонок
    for col_idx, col_name in enumerate(count_matrix.columns):
        worksheet.write(2, col_idx + 2, col_name, header_fmt if col_name != 'Grand Total' else grand_total_fmt)
        
    # Запис даних кількості
    for row_idx, row_name in enumerate(count_matrix.index):
        worksheet.write(row_idx + 3, 2, row_name, grand_total_fmt if row_name == 'Grand Total' else row_label_fmt)
        for col_idx, col_name in enumerate(count_matrix.columns):
            val = count_matrix.iloc[row_idx, col_idx]
            worksheet.write(row_idx + 3, col_idx + 3, val, grand_total_fmt if row_name == 'Grand Total' or col_name == 'Grand Total' else count_fmt)

    # --- ЗАПИС МАТРИЦІ ВІДСОТКІВ (Percentages) ---
    start_row = 11
    worksheet.merge_range(f'D{start_row+1}:H{start_row+1}', "Predictions", header_fmt)
    worksheet.merge_range(f'B{start_row+2}:B{start_row+6}', "Fact", row_label_fmt)
    
    for col_idx, col_name in enumerate(pct_matrix.columns):
        worksheet.write(start_row + 1, col_idx + 2, col_name, header_fmt if col_name != 'Grand Total' else grand_total_fmt)
        
    for row_idx, row_name in enumerate(pct_matrix.index):
        worksheet.write(row_idx + start_row + 2, 2, row_name, grand_total_fmt if row_name == 'Grand Total' else row_label_fmt)
        for col_idx, col_name in enumerate(pct_matrix.columns):
            val = pct_matrix.iloc[row_idx, col_idx]
            
            # Логіка застосування кольору
            if row_name == 'Grand Total' or col_name == 'Grand Total':
                cell_fmt = grand_total_fmt
            else:
                # Отримуємо букву кольору з нашої матриці логіки
                color_code = color_logic[row_idx][col_idx]
                cell_fmt = fmt_map[color_code]
                
            worksheet.write(row_idx + start_row + 2, col_idx + 3, val, cell_fmt)

    # --- ЗВЕДЕНА СТАТИСТИКА ЗОН (Справа) ---
    stats_row = start_row + 2
    worksheet.write(stats_row, 9, "", fmt_green)
    worksheet.write(stats_row, 10, zone_totals['G'], workbook.add_format({'num_format': '0.0%'}))
    
    worksheet.write(stats_row + 1, 9, "", fmt_yellow)
    worksheet.write(stats_row + 1, 10, zone_totals['Y'], workbook.add_format({'num_format': '0.0%'}))
    
    worksheet.write(stats_row + 2, 9, "", fmt_red)
    worksheet.write(stats_row + 2, 10, zone_totals['R'], workbook.add_format({'num_format': '0.0%'}))

    # Налаштування ширини колонок
    worksheet.set_column('A:A', 2)
    worksheet.set_column('B:B', 5)
    worksheet.set_column('C:C', 15)
    worksheet.set_column('D:H', 12)
    worksheet.set_column('I:I', 5)
    worksheet.set_column('J:J', 5)

    writer.close()
    print(f"Файл {output_filename} успішно згенеровано!")

# Виклик
# generate_peter_matrix_report(active_clients)