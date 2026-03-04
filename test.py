import pandas as pd
import numpy as np

# ==========================================
# 1. Завантаження даних та словники
# ==========================================
pilot_df = pd.read_excel('Pilot_clients.xlsx')
micro_df = pd.read_excel('Micro.xlsx')
small_df = pd.read_excel('Small.xlsx')
# Завантажуємо нову таблицю (зміни розширення на .xlsx якщо це ексель)
dash_df = pd.read_csv('dashboard_data.csv') 

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
dash_df['IDENTIFYCODE'] = clean_id(dash_df['IDENTIFYCODE'])

if pilot_df['PRIMARY'].dtype == object:
    pilot_df['PRIMARY'] = pilot_df['PRIMARY'].astype(str).str.replace('%', '').astype(float) / 100

# Формуємо множину D: всі ID з дашборду, які вважаються залученими
DASHBOARD_ACQUIRED_SET = set(dash_df['IDENTIFYCODE'].unique())

# ==========================================
# 3. Індикаторні функції (з урахуванням Dashboard)
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
# I_W
micro_df['Is_Worked'] = micro_df['Дата внесення змін'].notna().astype(int)

# I_S: (Статус Success АБО є в Дашборді) ТА (Є дата внесення змін, тобто реально працювали)
micro_in_dash = micro_df['Ідентифікаційний номер'].isin(DASHBOARD_ACQUIRED_SET)
micro_success_crm = micro_df['Category'] == 'Success'
micro_df['Is_Success'] = ((micro_success_crm | micro_in_dash) & micro_df['Дата внесення змін'].notna()).astype(int)

micro_flags = micro_df[['Ідентифікаційний номер', 'Is_Worked', 'Is_Success']].copy()

# --- Сегмент SMALL ---
small_df['Category'] = small_df['Результат дзвінка'].apply(
    lambda x: categorize_status(x, SMALL_SUCCESS, SMALL_IN_PROGRESS, SMALL_FAILED)
)
col_small_date = 'Юр.особа.Дата відкриття клієнта (по колонці юр. особа).Дата відкриття'

# I_W
small_df['Is_Worked'] = small_df['Дата створення'].notna().astype(int)

# I_S: (Статус Success ТА є дата відкриття) АБО (Є в Дашборді ТА взятий в роботу)
small_in_dash = small_df['Ідентифікаційний номер'].isin(DASHBOARD_ACQUIRED_SET)
small_success_crm = (small_df['Category'] == 'Success') & small_df[col_small_date].notna()

small_df['Is_Success'] = (small_success_crm | (small_in_dash & small_df['Дата створення'].notna())).astype(int)

small_flags = small_df[['Ідентифікаційний номер', 'Is_Worked', 'Is_Success']].copy()

# ==========================================
# 4. Злиття множин (max aggregation)
# ==========================================
all_flags = pd.concat([micro_flags, small_flags]).rename(columns={'Ідентифікаційний номер': 'IDENTIFYCODE'})
crm_combined = all_flags.groupby('IDENTIFYCODE', as_index=False).max()

master_df = pilot_df.merge(crm_combined, on='IDENTIFYCODE', how='left')
master_df['Is_Worked'] = master_df['Is_Worked'].fillna(0)
master_df['Is_Success'] = master_df['Is_Success'].fillna(0)
master_df['Is_Provided'] = 1

# ==========================================
# 5. Розрахунок метрик по місяцях
# ==========================================
metrics_dict = {}
months = sorted(master_df['MONTH'].dropna().unique())

for month in months:
    group = master_df[master_df['MONTH'] == month]
    t_provided = group['Is_Provided'].sum()
    t_worked = group['Is_Worked'].sum()
    t_success = group['Is_Success'].sum()
    
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
# 6. Розрахунок різниць (Дільт) та Експорт
# ==========================================
final_cols = []
final_data = {}

for i, month in enumerate(months):
    final_cols.append(month)
    final_data[month] = report_df[month]
    if i > 0:
        prev_month = months[i-1]
        diff_col = f"diff_{month}"
        final_cols.append(diff_col)
        final_data[diff_col] = report_df[month] - report_df[prev_month]

final_report_df = pd.DataFrame(final_data, columns=final_cols)

file_name = 'Model_Metrics_Report_Final.xlsx'
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
final_report_df.to_excel(writer, sheet_name='Metrics')

workbook = writer.book
worksheet = writer.sheets['Metrics']

# Стилі
fmt_head = workbook.add_format({'bold': True, 'bg_color': '#92D050', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
fmt_head_diff = workbook.add_format({'bold': True, 'bg_color': '#C4D79B', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
fmt_idx = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})
fmt_num = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'center'})
fmt_pct = workbook.add_format({'num_format': '0.00%', 'border': 1, 'align': 'center'})

fmt_num_diff_pos = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'center', 'font_color': '#00B050'})
fmt_num_diff_neg = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'center', 'font_color': '#FF0000'})
fmt_pct_diff_pos = workbook.add_format({'num_format': '0.00%', 'border': 1, 'align': 'center', 'font_color': '#00B050'})
fmt_pct_diff_neg = workbook.add_format({'num_format': '0.00%', 'border': 1, 'align': 'center', 'font_color': '#FF0000'})

worksheet.set_column('A:A', 35, fmt_idx)

for col_num, col_name in enumerate(final_report_df.columns, 1):
    is_diff = str(col_name).startswith('diff_')
    worksheet.set_column(col_num, col_num, 14 if is_diff else 16)
    worksheet.write(0, col_num, col_name, fmt_head_diff if is_diff else fmt_head)
    
    for row_num, metric_name in enumerate(final_report_df.index, 1):
        val = final_report_df.loc[metric_name, col_name]
        is_pct = 'propensity' in metric_name.lower() or 'rate' in metric_name.lower()
        
        if is_diff:
            if is_pct: cell_fmt = fmt_pct_diff_pos if val >= 0 else fmt_pct_diff_neg
            else: cell_fmt = fmt_num_diff_pos if val >= 0 else fmt_num_diff_neg
        else:
            cell_fmt = fmt_pct if is_pct else fmt_num
            
        if pd.isna(val): worksheet.write(row_num, col_num, "", cell_fmt)
        else: worksheet.write(row_num, col_num, val, cell_fmt)

writer.close()