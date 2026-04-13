import pandas as pd
import numpy as np

def create_business_report(input_df, output_filename='Business_Model_Report.xlsx'):
    df = input_df.copy()
    
    # 1. Очистка даних
    df['CONTRAGENTID'] = df['CONTRAGENTID'].replace('Missing value', np.nan)
    df['MONTHLY_INCOME'] = pd.to_numeric(df['MONTHLY_INCOME'].replace('Missing value', np.nan), errors='coerce')
    df['POTENTIAL_INCOME'] = pd.to_numeric(df['POTENTIAL_INCOME'], errors='coerce')

    # Фільтруємо існуючих клієнтів (тих, що мають і реальний, і потенційний дохід)
    eval_df = df.dropna(subset=['CONTRAGENTID', 'MONTHLY_INCOME', 'POTENTIAL_INCOME']).copy()

    # 2. ФОРМУВАННЯ ЛИСТА ТОЧНОСТІ (Як на скріншоті)
    # Залишаємо тільки потрібні колонки
    accuracy_df = eval_df[['IDENTIFYCODE', 'FIRM_NAME', 'FIRM_TYPE', 'MONTHLY_INCOME', 'POTENTIAL_INCOME']].copy()
    
    # Рахуємо абсолютну помилку
    accuracy_df['ABS_ERROR'] = (accuracy_df['MONTHLY_INCOME'] - accuracy_df['POTENTIAL_INCOME']).abs()
    
    # Сортуємо за найбільшою помилкою, щоб аномалії були зверху (як на скріншоті)
    accuracy_df = accuracy_df.sort_values(by='ABS_ERROR', ascending=False).reset_index(drop=True)

    # 3. ЗВЕДЕНІ ТАБЛИЦІ (З попереднього кроку)
    pivot_division = df.groupby(['DIVISION_CODE', 'DIVISION_NAME']).agg(
        Кількість_компаній=('IDENTIFYCODE', 'count'),
        Середній_Потенціал=('POTENTIAL_INCOME', 'mean'),
        Сумарний_Потенціал=('POTENTIAL_INCOME', 'sum')
    ).sort_values(by='Середній_Потенціал', ascending=False).reset_index()

    pivot_firm_type = df.groupby('FIRM_TYPE').agg(
        Кількість_компаній=('IDENTIFYCODE', 'count'),
        Середній_Потенціал=('POTENTIAL_INCOME', 'mean'),
        Сумарний_Потенціал=('POTENTIAL_INCOME', 'sum')
    ).sort_values(by='Середній_Потенціал', ascending=False).reset_index()

    pivot_our_clients = eval_df.groupby('FIRM_TYPE').agg(
        Кількість_наших_клієнтів=('IDENTIFYCODE', 'count'),
        Середній_Реальний_Дохід=('MONTHLY_INCOME', 'mean'),
        Середній_Потенціал=('POTENTIAL_INCOME', 'mean')
    ).sort_values(by='Кількість_наших_клієнтів', ascending=False).reset_index()

    # 4. ЗАПИС У EXCEL
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        money_fmt = workbook.add_format({'num_format': '#,##0', 'align': 'right'}) # Без копійок, як на скріншоті
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white', 'border': 1})
        title_fmt = workbook.add_format({'bold': True, 'font_size': 14})

        # --- Лист 1: Точність (Детальний аналіз помилок) ---
        accuracy_df.to_excel(writer, sheet_name='Точність Моделі', index=False)
        worksheet_accuracy = writer.sheets['Точність Моделі']
        
        # Застосування автофільтру та закріплення верхнього рядка
        worksheet_accuracy.autofilter(0, 0, len(accuracy_df), len(accuracy_df.columns) - 1)
        worksheet_accuracy.freeze_panes(1, 0)
        
        # Форматування колонок для листа точнісінько як на екрані
        worksheet_accuracy.set_column('A:A', 15) # IDENTIFYCODE
        worksheet_accuracy.set_column('B:B', 40) # FIRM_NAME
        worksheet_accuracy.set_column('C:C', 15) # FIRM_TYPE
        worksheet_accuracy.set_column('D:F', 20, money_fmt) # Грошові значення з форматом
        
        # Стилізація заголовків
        for col_num, value in enumerate(accuracy_df.columns.values):
            worksheet_accuracy.write(0, col_num, value, header_fmt)

        # --- Лист 2: Бізнес Аналітика ---
        worksheet_pivot = workbook.add_worksheet('Бізнес Аналітика')
        writer.sheets['Бізнес Аналітика'] = worksheet_pivot
        
        row_idx = 0
        worksheet_pivot.write_string(row_idx, 0, "1. Середня та загальна прибутковість за Сегментами (FIRM_TYPE)", title_fmt)
        row_idx += 1
        pivot_firm_type.to_excel(writer, sheet_name='Бізнес Аналітика', startrow=row_idx, index=False)
        for col_num, value in enumerate(pivot_firm_type.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)
        row_idx += len(pivot_firm_type) + 3
        
        worksheet_pivot.write_string(row_idx, 0, "2. Розподіл ІСНУЮЧИХ клієнтів банку", title_fmt)
        row_idx += 1
        pivot_our_clients.to_excel(writer, sheet_name='Бізнес Аналітика', startrow=row_idx, index=False)
        for col_num, value in enumerate(pivot_our_clients.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)
        row_idx += len(pivot_our_clients) + 3

        worksheet_pivot.write_string(row_idx, 0, "3. Рейтинг Індустрій (DIVISION)", title_fmt)
        row_idx += 1
        pivot_division.to_excel(writer, sheet_name='Бізнес Аналітика', startrow=row_idx, index=False)
        for col_num, value in enumerate(pivot_division.columns.values):
            worksheet_pivot.write(row_idx, col_num, value, header_fmt)

        worksheet_pivot.set_column('A:B', 35)
        worksheet_pivot.set_column('C:E', 20, workbook.add_format({'num_format': '#,##0.00', 'align': 'right'}))

        # --- Лист 3: Сирі дані ---
        df.to_excel(writer, sheet_name='База (Сирі Дані)', index=False)
        worksheet_raw = writer.sheets['База (Сирі Дані)']
        worksheet_raw.autofilter(0, 0, len(df), len(df.columns) - 1)
        worksheet_raw.freeze_panes(1, 0)
        
        for col_num, col_name in enumerate(df.columns):
            worksheet_raw.set_column(col_num, col_num, 15) 
            if 'POTENTIAL' in col_name or 'INCOME' in col_name:
                worksheet_raw.set_column(col_num, col_num, 20, workbook.add_format({'num_format': '#,##0.00', 'align': 'right'}))

    print(f"Звіт успішно згенеровано: {output_filename}")