import pandas as pd

def generate_full_hnwi_excel(df_inf, client_results, portrait_df_numeric, top_leads, output_path="HNWI_Business_Report_Full.xlsx"):
    """
    Генерує фінальний Excel-файл з трьома листами:
    1. Business Insights (Дашборд з ліфтами)
    2. Top Leads (Для продажів)
    3. Raw Info (Сирі дані по авто з флагом)
    """
    
    # Ініціалізація xlsxwriter
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    workbook = writer.book
    
    # --- 1. Створюємо формати (Стилі) ---
    header_fmt = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center',
        'bg_color': '#1F4E78', 'font_color': 'white', 'border': 1
    })
    float_fmt = workbook.add_format({'num_format': '#,##0.00', 'border': 1})
    pct_fmt = workbook.add_format({'num_format': '0.00%', 'border': 1})
    lift_fmt = workbook.add_format({'num_format': '0.00"x"', 'border': 1, 'align': 'center'})
    border_fmt = workbook.add_format({'border': 1})
    highlight_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'bold': True})
    
    # ==========================================
    # ЛИСТ 1: Business Insights (Дашборд)
    # ==========================================
    portrait_df_numeric.to_excel(writer, sheet_name='Business Insights', index=False)
    ws_insights = writer.sheets['Business Insights']
    
    # Форматування заголовків
    for col_num, value in enumerate(portrait_df_numeric.columns.values):
        ws_insights.write(0, col_num, value, header_fmt)
        
    # Форматування колонок
    ws_insights.set_column('A:A', 35, border_fmt) # Назва ознаки (зробили ширше)
    ws_insights.set_column('B:C', 20, float_fmt)  # Середні значення
    ws_insights.set_column('D:D', 15, lift_fmt)   # Lift
    
    # Колірна шкала для Lift
    ws_insights.conditional_format(1, 3, len(portrait_df_numeric), 3, {
        'type': '3_color_scale',
        'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B'
    })
    
    # Додаємо глобальні метрики праворуч
    hit_rate = client_results['is_potential_hnwi'].sum() / len(client_results)
    
    ws_insights.write('G2', 'Глобальна статистика', header_fmt)
    ws_insights.write('G3', 'Всього унікальних клієнтів', border_fmt)
    ws_insights.write('H3', len(client_results), float_fmt) # float_fmt додасть роздільники тисяч
    ws_insights.write('G4', 'Знайдено потенційних HNWI', border_fmt)
    ws_insights.write('H4', len(top_leads), float_fmt)
    ws_insights.write('G5', 'Hit Rate (Конверсія)', border_fmt)
    ws_insights.write('H5', hit_rate, pct_fmt)
    
    ws_insights.set_column('G:G', 30)
    ws_insights.set_column('H:H', 15)

    # ==========================================
    # ЛИСТ 2: Top Leads (Для продажів)
    # ==========================================
    top_leads.to_excel(writer, sheet_name='Top Leads', index=False)
    ws_leads = writer.sheets['Top Leads']
    
    for col_num, value in enumerate(top_leads.columns.values):
        ws_leads.write(0, col_num, value, header_fmt)
        
    ws_leads.set_column('A:Z', 18, border_fmt) # Базова ширина
    
    # Якщо є колонка з ймовірністю, додаємо DataBar
    if 'max_hnwi_prob' in top_leads.columns:
        prob_idx = top_leads.columns.get_loc('max_hnwi_prob')
        ws_leads.conditional_format(1, prob_idx, len(top_leads), prob_idx, {
            'type': 'data_bar', 'bar_color': '#63BE7B'
        })
        ws_leads.set_column(prob_idx, prob_idx, 18, pct_fmt)
        
    ws_leads.autofilter(0, 0, len(top_leads), len(top_leads.columns) - 1)

    # ==========================================
    # ЛИСТ 3: Raw Info (Сирі дані по авто)
    # ==========================================
    # Скидаємо індекси, щоб не було зайвих колонок в Excel
    df_inf_export = df_inf.copy()
    
    df_inf_export.to_excel(writer, sheet_name='Raw Info', index=False)
    ws_raw = writer.sheets['Raw Info']
    
    for col_num, value in enumerate(df_inf_export.columns.values):
        ws_raw.write(0, col_num, str(value), header_fmt)
        
    ws_raw.set_column('A:Z', 15) # Базова ширина колонок
    ws_raw.autofilter(0, 0, len(df_inf_export), len(df_inf_export.columns) - 1)
    
    # Підсвічуємо цільову колонку (де знайдено HNWI авто)
    target_col = 'is_hnwi_car' if 'is_hnwi_car' in df_inf_export.columns else 'is_hnwi'
    if target_col in df_inf_export.columns:
        target_idx = df_inf_export.columns.get_loc(target_col)
        
        # Якщо машина належить до HNWI (значення 1) - заливаємо клітинку світло-зеленим
        ws_raw.conditional_format(1, target_idx, len(df_inf_export), target_idx, {
            'type': 'cell', 'criteria': '==', 'value': 1,
            'format': highlight_fmt
        })

    # Збереження
    writer.close()
    print(f"✅ Звіт успішно збережено: {output_path}")

# --- ВИКЛИК ФУНКЦІЇ ---
# generate_full_hnwi_excel(df_inf, client_results, portrait_df_numeric, top_leads)