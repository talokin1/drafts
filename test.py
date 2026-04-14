import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# 1. Готуємо дані для прикладу (на основі ваших результатів)
# Аналіз Lift
lift_data = {
    'Ознака': ['Кількість авто', 'Сер. ціна портфеля ($)', 'Макс. ціна авто ($)', 'Володіння Luxury (%)'],
    'Середнє (Всі)': [1.29, 11206.48, 11689.43, 0.03],
    'Середнє (HNWI)': [1.86, 31547.78, 34187.06, 0.11],
    'Lift (X)': [1.44, 2.82, 2.92, 4.20]
}
df_lift = pd.DataFrame(lift_data)

# Топ ліди (приклад)
df_leads = pd.DataFrame({
    'MOBILEPHONE': ['380681853336', '380986826023', '380973805745', '380509040450', '380502560353'],
    'Ймовірність': [0.4985, 0.4985, 0.4985, 0.4985, 0.4985],
    'Кількість авто': [2, 33, 1, 1, 1],
    'Макс. ціна ($)': [72000, 53900, 48900, 41799, 35000],
    'Luxury Mark': ['Yes', 'No', 'Yes', 'No', 'No']
})

# 2. Створюємо Excel-книгу
wb = Workbook()
ws1 = wb.active
ws1.title = "📊 Загальний Огляд"
ws2 = wb.create_sheet("📈 Портрет та Lift")
ws3 = wb.create_sheet("💎 Топ-1000 Ліди")

# Стилі
header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
header_font = Font(color="FFFFFF", bold=True, size=12)
center_align = Alignment(horizontal="center", vertical="center")
border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

# ФУНКЦІЯ ДЛЯ СТИЛІЗАЦІЇ
def style_ws(ws, df, title):
    ws.append([title])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df.columns))
    ws.cell(row=1, column=1).font = Font(bold=True, size=14)
    ws.cell(row=1, column=1).alignment = center_align
    
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    
    for cell in ws[2]: # Header row
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
        
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=len(df.columns)):
        for cell in row:
            cell.border = border

# Наповнюємо листи
style_ws(ws2, df_lift, "Аналіз Lift (Порівняння сегменту з базою)")
style_ws(ws3, df_leads, "Топ пріоритетних клієнтів для опрацювання")

# Лист 1: Dashboard
ws1.append(["Звіт по сегменту потенційних HNWI клієнтів"])
ws1.append([""])
ws1.append(["Метрика", "Значення"])
stats = [
    ["Дата звіту", "14.04.2026"],
    ["Всього клієнтів у базі", 265426],
    ["Знайдено потенційних HNWI", 41458],
    ["Hit Rate (%)", "15.62%"],
    ["Середній Lift по вартості", "2.87x"]
]
for row in stats:
    ws1.append(row)

# Фінальне шліфування
for sheet in wb.sheetnames:
    ws = wb[sheet]
    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except: pass
        ws.column_dimensions[column].width = max_length + 5

wb.save("hnwi_potential_report_v1.xlsx")