import pandas as pd
import numpy as np

def generate_professional_hnwi_report(client_results, top_leads, features_to_profile, output_path='HNWI_Business_Report.xlsx'):
    """
    Скрипт генерує фінальний бізнес-звіт з урахуванням математичної коректності та стилізації.
    """
    
    # --- КРОК 1: Математична підготовка «Портрету» (Fixing the numeric issue) ---
    portrait_data = []
    # Важливо: використовуємо тільки ті дані, де клас вже визначений моделлю
    is_hnwi_mask = client_results['is_potential_hnwi'] == 1
    
    for feat in features_to_profile:
        mean_all = client_results[feat].mean()
        mean_hnwi = client_results[is_hnwi_mask][feat].mean()
        lift = (mean_hnwi / mean_all) if mean_all > 0 else 0
        
        portrait_data.append({
            'Ознака': feat,
            'Середнє (Всі)': mean_all,
            'Середнє (HNWI)': mean_hnwi,
            'Lift': lift
        })
    
    portrait_df_numeric = pd.DataFrame(portrait_data)
    
    # Розраховуємо глобальні метрики
    total_clients = len(client_results)
    hnwi_count = is_hnwi_mask.sum()
    hit_rate = hnwi_count / total_clients
    
    # --- КРОК 2: Ініціалізація Excel Writer ---
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    workbook = writer.book
    
    # Визначаємо стилі
    header_fmt = workbook.add_format({
        'bold': True, 'font_color': 'white', 'bg_color': '#1F4E78',
        'border': 1, 'valign': 'vcenter', 'align': 'center'
    })
    money_fmt = workbook.add_format({'num_format': '$#,##0.00', 'border': 1})
    pct_fmt = workbook.add_format({'num_format': '0.00%', 'border': 1})
    lift_fmt = workbook.add_format({'num_format': '0.00"x"', 'border': 1})
    num_fmt = workbook.add_format({'num_format': '#,##0.00', 'border': 1})
    border_fmt = workbook.add_format({'border': 1})
    title_fmt = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#1F4E78'})

    # --- ЛИСТ 1: Dashboard & Portrait ---
    sheet_name = 'Dashboard & Portrait'
    # Записуємо заголовок і метрики вручну для гарного вигляду
    ws_dash = workbook.add_worksheet(sheet_name)
    ws_dash.write('A1', 'БІЗНЕС-ПОРТРЕТ HNWI КЛІЄНТІВ (НА ОСНОВІ ML МОДЕЛІ)', title_fmt)
    
    # Блок глобальних метрик
    ws_dash.write('B3', 'Всього клієнтів', header_fmt)
    ws_dash.write('C3', 'Знайдено HNWI', header_fmt)
    ws_dash.write('D3', 'Hit Rate (Конверсія)', header_fmt)
    
    ws_dash.write('B4', total_clients, border_fmt)
    ws_dash.write('C4', hnwi_count, border_fmt)
    ws_dash.write('D4', hit_rate, pct_fmt)
    
    # Записуємо таблицю портрету (починаючи з 6-го рядка)
    start_row = 6
    for col_num, value in enumerate(portrait_df_numeric.columns.values):
        ws_dash.write(start_row, col_num, value, header_fmt)
        
    for row_idx, row in portrait_df_numeric.iterrows():
        ws_dash.write(start_row + row_idx + 1, 0, row['Ознака'], border_fmt)
        ws_dash.write(start_row + row_idx + 1, 1, row['Середнє (Всі)'], num_fmt)
        ws_dash.write(start_row + row_idx + 1, 2, row['Середнє (HNWI)'], num_fmt)
        ws_dash.write(start_row + row_idx + 1, 3, row['Lift'], lift_fmt)
        
    # Додаємо умовне форматування (колірну шкалу) для Lift
    ws_dash.conditional_format(start_row + 1, 3, start_row + len(portrait_df_numeric), 3, {
        'type': '3_color_scale',
        'min_color': "#F8696B", 'mid_color': "#FFEB84", 'max_color': "#63BE7B"
    })
    
    ws_dash.set_column('A:A', 25)
    ws_dash.set_column('B:D', 18)

    # --- ЛИСТ 2: Top Leads (Список для бізнесу) ---
    top_leads.to_excel(writer, sheet_name='Top Leads', index=False)
    ws_leads = writer.sheets['Top Leads']
    
    # Форматуємо заголовки
    for col_num, value in enumerate(top_leads.columns.values):
        ws_leads.write(0, col_num, value, header_fmt)
    
    # Додаємо "Data Bars" для ймовірності, щоб візуально виділити топ
    # Припустимо, Prob_HNWI — це 2-га колонка (індекс 1)
    ws_leads.conditional_format(1, 1, len(top_leads), 1, {
        'type': 'data_bar', 'bar_color': '#63BE7B'
    })
    
    ws_leads.set_column('A:A', 15) # Телефон
    ws_leads.set_column('B:B', 15, pct_fmt) # Probability
    ws_leads.set_column('C:Z', 15)
    ws_leads.autofilter(0, 0, len(top_leads), len(top_leads.columns) - 1)

    # Закриваємо файл
    writer.close()
    print(f"Звіт успішно збережено: {output_path}")

# ПРИКЛАД ВИКЛИКУ:
# generate_professional_hnwi_report(client_results, top_leads, ['cars_count', 'avg_price', 'max_price', 'has_luxury'])