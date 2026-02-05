import pandas as pd
import numpy as np

# ==========================================
# 0. ПЕРЕДУМОВА (Має бути вже виконано в пам'яті)
# ==========================================
# Ми припускаємо, що у тебе в пам'яті вже є змінна y_pred_stretched
# Це результат твого попереднього кроку: 
# y_pred_stretched = (y_pred_log - mean_pred) * scaling_factor + mean_true

# ==========================================
# 1. КОНВЕРТАЦІЯ ТА СТВОРЕННЯ БАЗОВОЇ ТАБЛИЦІ
# ==========================================

# Конвертуємо розтягнуті логарифми назад у гроші
pred_stretched_money = np.expm1(y_pred_stretched)
true_value_money = np.expm1(y_val_log)

# Створюємо dataframe результатів
# X_val.index містить IDENTIFYCODE (якщо ми зберегли його в індексі при спліті)
dist_results = pd.DataFrame({
    'IDENTIFYCODE': X_val.index,
    'True_Value': true_value_money,
    'Predicted': pred_stretched_money  # <--- ТУТ ТЕПЕР НОВИЙ, "РОЗТЯГНУТИЙ" ПРОГНОЗ
})

# ==========================================
# 2. ПІДТЯГУВАННЯ ДОДАТКОВИХ ДАНИХ (by_items)
# ==========================================
# Назва колонки, яку ти хочеш бачити як "by items"
LIABILITIES_COL = 'REVENUE_CUR' # <-- Перевір, чи це правильна назва колонки у df

# Мерджимо з оригінальним датасетом.
# Використовуємо right_index=True, бо в df зазвичай ID стоїть в індексі.
# Якщо ID - це колонка, зміни на: right_on='IDENTIFYCODE'
report_df = dist_results.merge(df[[LIABILITIES_COL]], left_on='IDENTIFYCODE', right_index=True, how='left')

# ==========================================
# 3. ФОРМУВАННЯ ФІНАЛЬНОЇ ТАБЛИЦІ
# ==========================================
final_report = pd.DataFrame()
final_report['CNUM'] = report_df['IDENTIFYCODE']
final_report['Y_target'] = report_df['Predicted']   # Stretched Model Prediction
final_report['by_items'] = report_df[LIABILITIES_COL] 
final_report['Fcst'] = report_df['True_Value']      # Fact

# Розрахунок різниці (Помилки)
final_report['Difs'] = (final_report['Y_target'] - final_report['Fcst']).abs()

# Округлення до 2 знаків
final_report = final_report.round(2)

# ==========================================
# 4. СТВОРЕННЯ БАКЕТІВ ТА ЗВЕДЕНОЇ ТАБЛИЦІ
# ==========================================
bins = [-1, 1000, 10000, 100000, 1000000, np.inf]
labels = ['0-1k', '1k-10k', '10k-100k', '100k-1M', '1M+']

final_report['Income_Range'] = pd.cut(final_report['Fcst'], bins=bins, labels=labels)

summary_table = final_report.groupby('Income_Range', observed=False)['Difs'].agg(
    Count='count',
    Mean_Error='mean',    # MAE
    Median_Error='median' # MedAE
).reset_index()

summary_table = summary_table.round(2)

# ==========================================
# 5. ЕКСПОРТ В EXCEL (З ФОРМАТУВАННЯМ)
# ==========================================
file_name = 'Distribution_Model_Report.xlsx' # Нове ім'я файлу, щоб не перезаписати старий

with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
    # 1. Запис детальної таблиці
    final_report.to_excel(writer, sheet_name='Report', startrow=1, header=False, index=False)
    
    # 2. Запис зведеної таблиці справа
    summary_start_col = 8 
    summary_table.to_excel(writer, sheet_name='Report', startrow=1, startcol=summary_start_col, header=False, index=False)

    # --- ДИЗАЙН ---
    workbook = writer.book
    worksheet = writer.sheets['Report']
    
    # Стилі
    header_format = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top',
        'fg_color': '#D7E4BC', 'border': 1
    })
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    
    # Заголовки (Детальна таблиця)
    for col_num, value in enumerate(final_report.columns):
        worksheet.write(0, col_num, value, header_format)
        
    # Заголовки (Зведена таблиця)
    headers_summary = ["Income Range (Analytics)", "Count", "Mean Error (MAE)", "Median Error"]
    for i, h in enumerate(headers_summary):
        worksheet.write(0, summary_start_col + i, h, header_format)

    # Ширина колонок
    worksheet.set_column('A:A', 15) 
    worksheet.set_column('B:E', 12, number_format) 
    worksheet.set_column('F:F', 15) 
    
    # Ширина для зведеної
    worksheet.set_column(summary_start_col, summary_start_col, 20)
    worksheet.set_column(summary_start_col+1, summary_start_col+3, 15, number_format)

    print(f"Звіт по Distribution Model збережено: {file_name}")
    print("\nПопередній перегляд зведеної таблиці:")
    print(summary_table)