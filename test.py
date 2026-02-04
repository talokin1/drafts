import pandas as pd
import numpy as np

# ==========================================
# 1. ПІДГОТОВКА ДАНИХ
# ==========================================

# Припустимо, validation_results вже існує (з попереднього кроку)
# validation_results має колонки: ['IDENTIFYCODE', 'True_Value', 'Predicted']

# Тобі треба підтягнути 'Liabilities' (Passives/Пасиви) з оригінального df
# Заміни 'TOTAL_LIABILITIES' на назву своєї колонки з пасивами в df
LIABILITIES_COL = 'REVENUE_CUR' # Для прикладу взяв Revenue, зміни на свою колонку пасивів!

# Мерджимо пасиви до результатів
# Ми використовуємо оригінальний df, де індекс або колонка - це IDENTIFYCODE
# Якщо в df IDENTIFYCODE в індексі:
report_df = validation_results.merge(df[[LIABILITIES_COL]], left_on='IDENTIFYCODE', right_index=True, how='left')
# Якщо в df IDENTIFYCODE це колонка:
# report_df = validation_results.merge(df[['IDENTIFYCODE', LIABILITIES_COL]], on='IDENTIFYCODE', how='left')

# ==========================================
# 2. ПЕРЕЙМЕНУВАННЯ ТА РОЗРАХУНКИ
# ==========================================
final_report = pd.DataFrame()
final_report['CNUM'] = report_df['IDENTIFYCODE']
final_report['Y_target'] = report_df['Predicted']  # Предикт моделі
final_report['by_items'] = report_df[LIABILITIES_COL] # Пасиви (Liabilities)
final_report['Fcst'] = report_df['True_Value']     # Реальне значення (Факт)

# Розрахунок абсолютної різниці (Difs)
final_report['Difs'] = (final_report['Y_target'] - final_report['Fcst']).abs()

# Округлення
final_report = final_report.round(2)

# ==========================================
# 3. СТВОРЕННЯ БАКЕТІВ (ГРУПУВАННЯ)
# ==========================================
# Розбиваємо клієнтів на групи за розміром реального доходу (Fcst)
# Можеш налаштувати межі (bins) під свої потреби
bins = [-1, 1000, 10000, 100000, 1000000, np.inf]
labels = ['0-1k', '1k-10k', '10k-100k', '100k-1M', '1M+']

final_report['Income_Range'] = pd.cut(final_report['Fcst'], bins=bins, labels=labels)

# ==========================================
# 4. ЗВЕДЕНА ТАБЛИЦЯ (ANALYTICS)
# ==========================================
summary_table = final_report.groupby('Income_Range', observed=False)['Difs'].agg(
    Count='count',
    Mean_Error='mean',
    Median_Error='median'
).reset_index()

# Додаємо форматування для читабельності
summary_table = summary_table.round(2)

# ==========================================
# 5. ЕКСПОРТ В EXCEL (З ФОРМАТУВАННЯМ)
# ==========================================
file_name = 'Model_Analysis_Report.xlsx'

with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
    # 1. Записуємо основну таблицю зліва
    final_report.to_excel(writer, sheet_name='Report', startrow=1, header=False, index=False)
    
    # 2. Записуємо зведену таблицю справа (наприклад, колонка I = 8)
    summary_start_col = 8 
    summary_table.to_excel(writer, sheet_name='Report', startrow=1, startcol=summary_start_col, header=False, index=False)

    # --- ФОРМАТУВАННЯ ---
    workbook = writer.book
    worksheet = writer.sheets['Report']
    
    # Стилі
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BC', # Світло-зелений як на скріні
        'border': 1
    })
    
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    
    # Заголовки для основної таблиці
    for col_num, value in enumerate(final_report.columns):
        worksheet.write(0, col_num, value, header_format)
        
    # Заголовки для зведеної таблиці
    worksheet.write(0, summary_start_col, "Income Range (Analytics)", header_format)
    worksheet.write(0, summary_start_col + 1, "Count", header_format)
    worksheet.write(0, summary_start_col + 2, "Mean Error (MAE)", header_format)
    worksheet.write(0, summary_start_col + 3, "Median Error", header_format)

    # Налаштування ширини колонок
    worksheet.set_column('A:A', 15) # CNUM
    worksheet.set_column('B:D', 12, number_format) # Y_target, by_items, Fcst
    worksheet.set_column('E:E', 12, number_format) # Difs
    worksheet.set_column('F:F', 15) # Income_Range
    
    # Ширина для зведеної таблиці
    worksheet.set_column(summary_start_col, summary_start_col, 20)
    worksheet.set_column(summary_start_col+1, summary_start_col+3, 15, number_format)

    print(f"Звіт успішно збережено у файл: {file_name}")
    print("\nПопередній перегляд зведеної таблиці:")
    print(summary_table)