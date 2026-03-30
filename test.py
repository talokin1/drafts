import pandas as pd
import numpy as np

def generate_business_excel(df, output_filename="Potential_Income_Report.xlsx"):
    # 1. Підготовка даних та розрахунок метрик
    df = df.copy()
    
    # Визначаємо, чи є компанія нашим клієнтом
    df['IS_CLIENT'] = df['CONTRAGENTID'].notna()
    
    # Розрахунок помилок для існуючих клієнтів
    # Використовуємо np.where, щоб уникнути ділення на нуль
    df['ABS_ERROR'] = np.where(df['IS_CLIENT'], abs(df['MONTHLY_INCOME'] - df['POTENTIAL_INCOME']), np.nan)
    df['ERROR_PCT'] = np.where(df['IS_CLIENT'] & (df['MONTHLY_INCOME'] > 0), 
                               df['ABS_ERROR'] / df['MONTHLY_INCOME'], np.nan)
    
    # Створення зведених таблиць (Pivot)
    # Зведення 1: Потенційний дохід за розміром бізнесу (FIRM_TYPE)
    pivot_firm_type = pd.pivot_table(df, 
                                     index='FIRM_TYPE', 
                                     values=['IDENTIFYCODE', 'POTENTIAL_INCOME', 'MONTHLY_INCOME'],
                                     aggfunc={'IDENTIFYCODE': 'count', 
                                              'POTENTIAL_INCOME': 'sum', 
                                              'MONTHLY_INCOME': 'sum'}).rename(columns={'IDENTIFYCODE': 'Кількість компаній'})
    
    # Зведення 2: Потенційний дохід за галуззю (KVED_DESCR)
    pivot_kved = pd.pivot_table(df, 
                                index='KVED_DESCR', 
                                values=['POTENTIAL_INCOME'],
                                aggfunc=['count', 'sum'])
    pivot_kved.columns = ['Кількість компаній', 'Сумарний потенціал']
    pivot_kved = pivot_kved.sort_values(by='Сумарний потенціал', ascending=False).head(15) # Топ-15 галузей
    
    # Дані для листа "Точність моделі" (лише клієнти)
    accuracy_df = df[df['IS_CLIENT']].copy()
    accuracy_df = accuracy_df[['IDENTIFYCODE', 'FIRM_NAME', 'FIRM_TYPE', 'MONTHLY_INCOME', 'POTENTIAL_INCOME', 'ABS_ERROR', 'ERROR_PCT']]
    accuracy_df = accuracy_df.sort_values(by='ERROR_PCT', ascending=False)

    # 2. Запис у Excel з форматуванням
    writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
    workbook = writer.book
    
    # Формати
    money_fmt = workbook.add_format({'num_format': '#,##0.00 ₴'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#1F497D', 'font_color': 'white', 'border': 1})
    
    # --- ЛИСТ 1: Сирі дані ---
    df.to_excel(writer, sheet_name='Raw_Data', index=False)
    worksheet1 = writer.sheets['Raw_Data']
    worksheet1.autofilter(0, 0, len(df), len(df.columns) - 1)
    worksheet1.freeze_panes(1, 0)
    
    # Форматування колонок (приблизно ширини колонок)
    worksheet1.set_column('A:B', 15)
    worksheet1.set_column('C:G', 18, money_fmt) # Колонки з потенціалом
    worksheet1.set_column('H:H', 30) # FIRM_NAME
    worksheet1.set_column('P:P', 18, money_fmt) # MONTHLY_INCOME
    
    # Застосування стилю заголовків
    for col_num, value in enumerate(df.columns.values):
        worksheet1.write(0, col_num, value, header_fmt)

    # --- ЛИСТ 2: Зведена аналітика ---
    pivot_firm_type.to_excel(writer, sheet_name='Summary_Dashboard', startrow=1, startcol=1)
    pivot_kved.to_excel(writer, sheet_name='Summary_Dashboard', startrow=len(pivot_firm_type) + 5, startcol=1)
    worksheet2 = writer.sheets['Summary_Dashboard']
    worksheet2.write(0, 1, "Аналітика за типом компанії", workbook.add_format({'bold': True, 'font_size': 14}))
    worksheet2.write(len(pivot_firm_type) + 4, 1, "Топ-15 галузей (КВЕД) за потенціалом", workbook.add_format({'bold': True, 'font_size': 14}))
    worksheet2.set_column('B:B', 35)
    worksheet2.set_column('C:E', 20, money_fmt)

    # --- ЛИСТ 3: Точність моделі ---
    accuracy_df.to_excel(writer, sheet_name='Model_Accuracy', index=False)
    worksheet3 = writer.sheets['Model_Accuracy']
    worksheet3.autofilter(0, 0, len(accuracy_df), len(accuracy_df.columns) - 1)
    worksheet3.freeze_panes(1, 0)
    
    worksheet3.set_column('A:A', 15)
    worksheet3.set_column('B:B', 30)
    worksheet3.set_column('C:C', 15)
    worksheet3.set_column('D:F', 20, money_fmt) # INCOME та ERROR
    worksheet3.set_column('G:G', 15, pct_fmt)   # ERROR_PCT
    
    # Умовне форматування для відсотків помилки (3-кольорова шкала: Зелений-Жовтий-Червоний)
    worksheet3.conditional_format(1, 6, len(accuracy_df), 6, {
        'type': '3_color_scale',
        'min_color': '#63BE7B', # Зелений для маленької помилки
        'mid_color': '#FFEB84',
        'max_color': '#F8696B', # Червоний для великої
        'min_type': 'num', 'min_value': 0.0,
        'mid_type': 'num', 'mid_value': 0.2, # 20% помилки - це жовта зона
        'max_type': 'num', 'max_value': 0.5  # >50% помилки - червона зона
    })
    
    for col_num, value in enumerate(accuracy_df.columns.values):
        worksheet3.write(0, col_num, value, header_fmt)

    writer.close()
    print(f"Файл {output_filename} успішно згенеровано!")

# Виклик функції (просто передайте сюди ваш існуючий датафрейм)
# generate_business_excel(df)