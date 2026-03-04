import pandas as pd
import numpy as np

# ==========================================
# 1. Завантаження даних та словники
# ==========================================
# Заміни назви файлів, якщо вони відрізняються
pilot_df = pd.read_excel('Pilot_clients.xlsx')
micro_df = pd.read_excel('Micro.xlsx')
small_df = pd.read_excel('Small.xlsx')

# --- Множини статусів MICRO ---
MICRO_SUCCESS = {"Рахунок відкрито", "Відкриття рахунку"}
MICRO_IN_PROGRESS = {"В роботі", "Необхідно подумати", "Передано на RM", "Зустріч на відділені"}
MICRO_FAILED = {"Відмова", "Новий", "Неактуальні контактні дані", "Недодзвон"}

# --- Множини статусів SMALL ---
SMALL_SUCCESS = {"Відкриття рахунку"}
SMALL_IN_PROGRESS = {
    "Клієнт зацікавлений", 
    "Клієнт зацікавлений. Потрібен дзвінок/зустріч з RM", 
    "Консультацію не проведено. Потрібен додатковий дзвінок", 
    "Консультацію проведено. Потрібен додатковий дзвінок", 
    "Консультацію проведено"
}
SMALL_FAILED = {
    "Консультацію проведено. Відмова клієнта", "Консультацію проведено, відмова клієнта", 
    "Клієнт не зацікавлений", "Клієнт відмовився від зустрічі", "Відмова клієнта", 
    "Неможливо зв'язатись з клієнтом. Номер телефону недійсний", 
    "Неможливо зв'язатись з клієнтом. Не відповідає на дзвінок", 
    "Неможливо дізнатись актуальну Контактну особу клієнта", 
    "Не актуальні контактні дані", "Контакт не актуальний", 
    "Консультацію проведено. Не відповідає сегментації.", 
    "Відмова ОТР Банку (framework)", "Відмова ОТР Банку (фін.моніторинг)", 
    "Відмова ОТР Банку (Клієнт не відповідає вимогам)", "Відмова ОТР Банку (фін. стан)", 
    "Відмова ОТР Банку (Бенефіціар - громадянин РФ)", "Виконано", "Виконано з помилками"
}

# ==========================================
# 2. Передобробка: очищення ключів
# ==========================================
def clean_id(series):
    return series.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

pilot_df['IDENTIFYCODE'] = clean_id(pilot_df['IDENTIFYCODE'])
micro_df['Ідентифікаційний номер'] = clean_id(micro_df['Ідентифікаційний номер'])
small_df['Ідентифікаційний номер'] = clean_id(small_df['Ідентифікаційний номер'])

if pilot_df['PRIMARY'].dtype == object:
    pilot_df['PRIMARY'] = pilot_df['PRIMARY'].astype(str).str.replace('%', '').astype(float) / 100

# ==========================================
# 3. Індикаторні функції з урахуванням словників і дат
# ==========================================
def categorize_status(val, success_set, prog_set, fail_set):
    val = str(val).strip()
    if val in success_set: return 'Success'
    if val in prog_set: return 'In Process'
    if val in fail_set: return 'Failed'
    return 'Unknown'

# --- Сегмент MICRO ---
micro_df['Category'] = micro_df['Тематика дзвінка'].apply(
    lambda x: categorize_status(x, MICRO_SUCCESS, MICRO_IN_PROGRESS, MICRO_FAILED)
)
# I_W = 1, якщо є дата внесення змін
micro_df['Is_Worked'] = micro_df['Дата внесення змін'].notna().astype(int)
# I_S = 1, якщо статус Success І є дата внесення змін
micro_df['Is_Success'] = ((micro_df['Category'] == 'Success') & micro_df['Дата внесення змін'].notna()).astype(int)
micro_flags = micro_df[['Ідентифікаційний номер', 'Is_Worked', 'Is_Success']].copy()

# --- Сегмент SMALL ---
small_df['Category'] = small_df['Результат дзвінка'].apply(
    lambda x: categorize_status(x, SMALL_SUCCESS, SMALL_IN_PROGRESS, SMALL_FAILED)
)
col_small_date = 'Юр.особа.Дата відкриття клієнта (по колонці юр. особа).Дата відкриття'
# I_W = 1, якщо є дата створення
small_df['Is_Worked'] = small_df['Дата створення'].notna().astype(int)
# I_S = 1, якщо статус Success І є дата відкриття
small_df['Is_Success'] = ((small_df['Category'] == 'Success') & small_df[col_small_date].notna()).astype(int)
small_flags = small_df[['Ідентифікаційний номер', 'Is_Worked', 'Is_Success']].copy()

# ==========================================
# 4. Злиття множин (max aggregation)
# ==========================================
all_flags = pd.concat([micro_flags, small_flags]).rename(columns={'Ідентифікаційний номер': 'IDENTIFYCODE'})
# Якщо клієнт є в обох базах, операція max() реалізує логічне АБО (1 ∨ 0 = 1)
crm_combined = all_flags.groupby('IDENTIFYCODE', as_index=False).max()

# Об'єднання з генеральною сукупністю наданих лідів
master_df = pilot_df.merge(crm_combined, on='IDENTIFYCODE', how='left')
master_df['Is_Worked'] = master_df['Is_Worked'].fillna(0)
master_df['Is_Success'] = master_df['Is_Success'].fillna(0)
master_df['Is_Provided'] = 1

# ==========================================
# 5. Розрахунок метрик по місяцях (Батчах)
# ==========================================
metrics_dict = {}
months = sorted(master_df['MONTH'].dropna().unique())

for month in months:
    group = master_df[master_df['MONTH'] == month]
    t_provided = group['Is_Provided'].sum()
    t_worked = group['Is_Worked'].sum()
    t_success = group['Is_Success'].sum()
    
    # Умовні математичні сподівання схильності
    prop_success = group[group['Is_Success'] == 1]['PRIMARY'].mean()
    prop_failed = group[group['Is_Success'] == 0]['PRIMARY'].mean()
    
    conversion = (t_success / t_worked) if t_worked > 0 else 0
    
    metrics_dict[month] = {
        '1. Total potential clients': t_provided,
        '2. Clients taken into work': t_worked,
        '3. Acquired clients (Success)': t_success,
        '4. Avg propensity of acquired': prop_success if pd.notna(prop_success) else 0.0,
        '5. Avg propensity of not acquired': prop_failed if pd.notna(prop_failed) else 0.0,
        '6. Conversion Rate (from worked)': conversion
    }

report_df = pd.DataFrame(metrics_dict)

# ==========================================
# 6. Розрахунок різниць (Дільт) між місяцями
# ==========================================
# Створюємо новий DataFrame, де будемо чергувати: Місяць -> Різниця
final_cols = []
final_data = {}

for i, month in enumerate(months):
    final_cols.append(month)
    final_data[month] = report_df[month]
    
    if i > 0:
        prev_month = months[i-1]
        diff_col_name = f"diff_{month}"
        final_cols.append(diff_col_name)
        # Рахуємо дельту (x_t - x_{t-1})
        final_data[diff_col_name] = report_df[month] - report_df[prev_month]

# Збираємо фінальну таблицю з правильним порядком колонок
final_report_df = pd.DataFrame(final_data, columns=final_cols)

# ==========================================
# 7. Експорт та форматування Excel (xlsxwriter)
# ==========================================
file_name = 'Model_Metrics_Report_Final.xlsx'
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
final_report_df.to_excel(writer, sheet_name='Metrics')

workbook = writer.book
worksheet = writer.sheets['Metrics']

# --- Визначення стилів ---
format_header = workbook.add_format({
    'bold': True, 'bg_color': '#92D050', 'border': 1, 'align': 'center', 'valign': 'vcenter'
})
format_header_diff = workbook.add_format({
    'bold': True, 'bg_color': '#C4D79B', 'border': 1, 'align': 'center', 'valign': 'vcenter' # Трохи світліший зелений
})
format_idx = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})

# Формати для базових колонок
format_num = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'center'})
format_pct = workbook.add_format({'num_format': '0.00%', 'border': 1, 'align': 'center'})

# Формати для колонок diff (з кольорами для додатних і від'ємних значень)
format_num_diff_pos = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'center', 'font_color': '#00B050'})
format_num_diff_neg = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'center', 'font_color': '#FF0000'})
format_pct_diff_pos = workbook.add_format({'num_format': '0.00%', 'border': 1, 'align': 'center', 'font_color': '#00B050'})
format_pct_diff_neg = workbook.add_format({'num_format': '0.00%', 'border': 1, 'align': 'center', 'font_color': '#FF0000'})

# --- Застосування стилів ---
worksheet.set_column('A:A', 35, format_idx) # Перша колонка з назвами

for col_num, col_name in enumerate(final_report_df.columns, 1):
    is_diff_col = str(col_name).startswith('diff_')
    
    # Ширина колонки
    worksheet.set_column(col_num, col_num, 14 if is_diff_col else 16)
    
    # Заголовок
    header_format = format_header_diff if is_diff_col else format_header
    worksheet.write(0, col_num, col_name, header_format)
    
    # Заповнення рядків
    for row_num, metric_name in enumerate(final_report_df.index, 1):
        val = final_report_df.loc[metric_name, col_name]
        is_pct_metric = 'propensity' in metric_name.lower() or 'rate' in metric_name.lower()
        
        # Визначаємо потрібний формат залежно від типу метрики та знаку (для diff)
        if is_diff_col:
            if is_pct_metric:
                cell_format = format_pct_diff_pos if val >= 0 else format_pct_diff_neg
            else:
                cell_format = format_num_diff_pos if val >= 0 else format_num_diff_neg
        else:
            cell_format = format_pct if is_pct_metric else format_num
            
        # Записуємо значення (якщо val == NaN, пишемо порожньо)
        if pd.isna(val):
            worksheet.write(row_num, col_num, "", cell_format)
        else:
            worksheet.write(row_num, col_num, val, cell_format)

writer.close()
print(f"Готово! Звіт збережено у файл {file_name}")