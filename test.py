import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

def append_peters_classification(eval_df, output_filename='Business_Model_Report.xlsx'):
    df = eval_df.copy()
    
    # 1. НАЛАШТУВАННЯ КЛАСІВ ТА ГРАНИЦЬ (БІЗНЕС-ЛОГІКА)
    # УВАГА: Заміни ці цифри на свої реальні пороги (bins) для MONTHLY_INCOME!
    # Наприклад: до 10k - Hopeless, до 50k - Bad Quality і т.д.
    bins = [-np.inf, 10000, 50000, 150000, 500000, np.inf]
    categories = ['Total Hopeless', 'Bad Quality', 'Grey Zone', 'Good Quality', 'Fantastic']
    
    # Тільки для клієнтів, у яких є і факт, і прогноз
    df = df.dropna(subset=['MONTHLY_INCOME', 'POTENTIAL_INCOME'])
    
    # Присвоюємо класи на основі порогів
    df['Fact_Class'] = pd.cut(df['MONTHLY_INCOME'], bins=bins, labels=categories)
    df['Pred_Class'] = pd.cut(df['POTENTIAL_INCOME'], bins=bins, labels=categories)
    
    # Робимо їх категоріальними, щоб матриця завжди була 5x5 (навіть якщо клас порожній)
    df['Fact_Class'] = pd.Categorical(df['Fact_Class'], categories=categories, ordered=True)
    df['Pred_Class'] = pd.Categorical(df['Pred_Class'], categories=categories, ordered=True)

    # 2. СТВОРЕННЯ МАТРИЦЬ (Абсолютна та Відносна)
    # Матриця в штуках (кількість клієнтів)
    matrix_abs = pd.crosstab(df['Fact_Class'], df['Pred_Class'], dropna=False)
    
    # Матриця у відсотках (від загальної кількості)
    matrix_pct = pd.crosstab(df['Fact_Class'], df['Pred_Class'], dropna=False, normalize='all')

    # 3. ЗАПИС У ІСНУЮЧИЙ EXCEL ТА ФОРМАТУВАННЯ ЗОН
    try:
        book = load_workbook(output_filename)
    except FileNotFoundError:
        print(f"Помилка: Файл {output_filename} не знайдено. Спочатку згенеруй основний звіт.")
        return

    with pd.ExcelWriter(output_filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        sheet_name = "Peter's Classification"
        
        # Записуємо таблиці. Відступаємо місце між ними.
        matrix_abs.to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=1)
        matrix_pct.to_excel(writer, sheet_name=sheet_name, startrow=10, startcol=1)
        
        worksheet = writer.sheets[sheet_name]
        
        # Додаємо заголовки
        worksheet.cell(row=1, column=2, value="Matrix: Absolute Count").font = Font(bold=True, size=12)
        worksheet.cell(row=2, column=1, value="FACT \ PRED").font = Font(bold=True)
        worksheet.cell(row=9, column=2, value="Matrix: Percentages (%)").font = Font(bold=True, size=12)
        worksheet.cell(row=10, column=1, value="FACT \ PRED").font = Font(bold=True)

        # Кольори для зон
        green_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
        yellow_fill = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
        red_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
        
        thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

        # Функція для розфарбовування матриці (логіка відстані |i - j|)
        def format_matrix(start_r, start_c, is_pct=False):
            for i in range(len(categories)):
                for j in range(len(categories)):
                    cell = worksheet.cell(row=start_r + i + 1, column=start_c + j + 1)
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal='center')
                    
                    if is_pct:
                        cell.number_format = '0.00%' # Формат відсотків
                        
                    distance = abs(i - j)
                    if distance <= 1:
                        cell.fill = green_fill
                    elif distance == 2:
                        cell.fill = yellow_fill
                    else:
                        cell.fill = red_fill
                        
            # Форматування назв рядків і стовпців
            for idx in range(len(categories)):
                worksheet.cell(row=start_r + idx + 1, column=start_c).font = Font(bold=True) # Рядки
                worksheet.cell(row=start_r, column=start_c + idx + 1).font = Font(bold=True) # Стовпці
        
        # Застосовуємо форматування
        format_matrix(start_r=3, start_c=2, is_pct=False)
        format_matrix(start_r=11, start_c=2, is_pct=True)

        # Регулюємо ширину колонок
        worksheet.column_dimensions['B'].width = 15
        for col in ['C', 'D', 'E', 'F', 'G']:
            worksheet.column_dimensions[col].width = 18

    print(f"Аркуш 'Peter's Classification' успішно додано до {output_filename}")

# Приклад виклику:
# append_peters_classification(df, "Business_Model_Report.xlsx")