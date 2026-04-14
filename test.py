import pandas as pd

def generate_hnwi_excel_report(client_results, portrait_df_numeric, top_leads, hit_rate, output_path="HNWI_Business_Report.xlsx"):
    """
    Генерує відформатований Excel-файл для презентації стейкхолдерам.
    """
    # Створюємо Excel writer з використанням xlsxwriter
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    workbook = writer.book

    # --- 1. Створення форматів (Стилістичний словник) ---
    format_header = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 
        'fg_color': '#1F4E78', 'font_color': 'white', 'border': 1
    })
    format_money = workbook.add_format({'num_format': '$#,##0', 'border': 1})
    format_float = workbook.add_format({'num_format': '0.00', 'border': 1})
    format_pct = workbook.add_format({'num_format': '0.0%', 'border': 1})
    format_border = workbook.add_format({'border': 1})
    
    # --- Лист 1: Бізнес-портрет та Інсайти ---
    portrait_df_numeric.to_excel(writer, sheet_name='Business Insights', index=False)
    worksheet_insights = writer.sheets['Business Insights']
    
    # Форматуємо заголовки
    for col_num, value in enumerate(portrait_df_numeric.columns.values):
        worksheet_insights.write(0, col_num, value, format_header)
        
    # Додаємо умовне форматування для колонки "Lift" (припустимо, вона 4-та, індекс 3)
    # Це інтуїтивно покаже, які фічі "вистрілюють" найсильніше
    worksheet_insights.conditional_format('D2:D10', {
        'type': '3_color_scale',
        'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B'
    })
    
    # Додаємо загальну статистику збоку
    worksheet_insights.write('G2', 'Загальна статистика бази', format_header)
    worksheet_insights.write('G3', 'Всього унікальних клієнтів', format_border)
    worksheet_insights.write('H3', len(client_results), format_border)
    worksheet_insights.write('G4', 'Знайдено HNWI лідів', format_border)
    worksheet_insights.write('H4', len(top_leads), format_border)
    worksheet_insights.write('G5', 'Hit Rate (Конверсія)', format_border)
    worksheet_insights.write('H5', hit_rate, format_pct)
    
    worksheet_insights.set_column('A:A', 20)
    worksheet_insights.set_column('B:E', 15)
    worksheet_insights.set_column('G:G', 25)

    # --- Лист 2: Top Leads (Для продажів) ---
    top_leads.to_excel(writer, sheet_name='Top Leads', index=False)
    worksheet_leads = writer.sheets['Top Leads']
    
    for col_num, value in enumerate(top_leads.columns.values):
        worksheet_leads.write(0, col_num, value, format_header)
        
    # Застосовуємо формати до колонок (індекси залежать від твого датафрейму)
    # Припустимо: MOBILEPHONE (0), cars_count (1), max_hnwi_prob (2), is_hnwi (3), avg_price (4), max_price (5), has_luxury (6)
    worksheet_leads.set_column('A:A', 15, format_border) # Телефон
    worksheet_leads.set_column('B:B', 12, format_border) # Кількість авто
    worksheet_leads.set_column('C:C', 18, format_pct)    # Ймовірність HNWI
    worksheet_leads.set_column('D:D', 15, format_border) # Флаг
    worksheet_leads.set_column('E:F', 15, format_money)  # Ціни в $
    worksheet_leads.set_column('G:G', 12, format_border) # Luxury флаг

    # Градієнт для ймовірності, щоб сейлзи бачили "найгарячіших"
    worksheet_leads.conditional_format('C2:C10000', {
        'type': 'data_bar', 'bar_color': '#63BE7B'
    })

    # Додаємо автофільтр для зручності
    worksheet_leads.autofilter(0, 0, len(top_leads), len(top_leads.columns) - 1)

    # --- Лист 3: Raw Client Data ---
    client_results.to_excel(writer, sheet_name='Raw Aggregated Data', index=False)
    worksheet_raw = writer.sheets['Raw Aggregated Data']
    
    for col_num, value in enumerate(client_results.columns.values):
        worksheet_raw.write(0, col_num, value, format_header)
    worksheet_raw.set_column('A:Z', 15)

    # Зберігаємо
    writer.close()
    print(f"Звіт успішно збережено у {output_path}")

# Виклик функції:
# generate_hnwi_excel_report(client_results, portrait_df_numeric, top_leads, hit_rate)