import pandas as pd
import numpy as np

# ==========================================
# 1. Завантаження даних
# ==========================================
pilot_df = pd.read_excel('Pilot_clients.xlsx')
micro_df = pd.read_excel('Micro.xlsx')
small_df = pd.read_excel('Small.xlsx')

# ==========================================
# 2. Передобробка: очищення ключів та ймовірностей
# ==========================================
# ІПН/ЄДРПОУ часто гублять нулі на початку, тому переводимо в рядки і чистимо
pilot_df['IDENTIFYCODE'] = pilot_df['IDENTIFYCODE'].astype(str).str.strip()
micro_df['Ідентифікаційний номер'] = micro_df['Ідентифікаційний номер'].astype(str).str.strip()
small_df['Ідентифікаційний номер'] = small_df['Ідентифікаційний номер'].astype(str).str.strip()

# Переводимо PRIMARY у числовий формат (якщо раптом там є символ '%')
if pilot_df['PRIMARY'].dtype == object:
    pilot_df['PRIMARY'] = pilot_df['PRIMARY'].astype(str).str.replace('%', '').astype(float) / 100

# ==========================================
# 3. Маркування статусів (Micro та Small) з новими множинами
# ==========================================

# --- Визначаємо множини для MICRO ---
MICRO_SUCCESS = {
    "Рахунок відкрито", 
    "Відкриття рахунку"
}

MICRO_IN_PROGRESS = {
    "В роботі", 
    "Необхідно подумати", 
    "Передано на RM", 
    "Зустріч на відділені"
}

MICRO_FAILED = {
    "Відмова", 
    "Новий", 
    "Неактуальні контактні дані", 
    "Недодзвон"
}

# --- Визначаємо множини для SMALL ---
SMALL_SUCCESS = {
    "Відкриття рахунку"
}

SMALL_IN_PROGRESS = {
    "Клієнт зацікавлений", 
    "Клієнт зацікавлений. Потрібен дзвінок/зустріч з RM", 
    "Консультацію не проведено. Потрібен додатковий дзвінок", 
    "Консультацію проведено. Потрібен додатковий дзвінок", 
    "Консультацію проведено"
}

SMALL_FAILED = {
    "Консультацію проведено. Відмова клієнта", 
    "Консультацію проведено, відмова клієнта", 
    "Клієнт не зацікавлений", 
    "Клієнт відмовився від зустрічі", 
    "Відмова клієнта", 
    "Неможливо зв'язатись з клієнтом. Номер телефону недійсний", 
    "Неможливо зв'язатись з клієнтом. Не відповідає на дзвінок", 
    "Неможливо дізнатись актуальну Контактну особу клієнта", 
    "Не актуальні контактні дані", 
    "Контакт не актуальний", 
    "Консультацію проведено. Не відповідає сегментації.", 
    "Відмова OTP Банку (framework)", 
    "Відмова OTP Банку (фін.моніторинг)", 
    "Відмова OTP Банку (Клієнт не відповідає вимогам)", 
    "Відмова OTP Банку (фін. стан)", 
    "Відмова OTP Банку (Бенефіціар - громадянин РФ)", 
    "Виконано", 
    "Виконано з помилками"
}

# --- Універсальна функція мапінгу ---
def map_status(val, success_set, in_progress_set, failed_set):
    # Очищуємо значення: переводимо в рядок і прибираємо зайві пробіли
    val = str(val).strip() 
    
    if val in success_set:
        return 'Success'
    elif val in in_progress_set:
        return 'In Process'
    elif val in failed_set:
        return 'Failed'
    else:
        # Важливий запобіжник: якщо бізнес додасть новий статус, ми його не загубимо, 
        # а побачимо як Unknown
        return 'Unknown' 

# Застосовуємо мапінг. 
# Вважаємо, що для Micro колонка називається "Тематика дзвінка", для Small - "Результат дзвінка"
micro_df['Unified_Status'] = micro_df['Тематика дзвінка'].apply(
    lambda x: map_status(x, MICRO_SUCCESS, MICRO_IN_PROGRESS, MICRO_FAILED)
)

small_df['Unified_Status'] = small_df['Результат дзвінка'].apply(
    lambda x: map_status(x, SMALL_SUCCESS, SMALL_IN_PROGRESS, SMALL_FAILED)
)

# --- Об'єднання баз (з урахуванням пріоритетів) ---
micro_cut = micro_df[['Ідентифікаційний номер', 'Unified_Status']].copy()
small_cut = small_df[['Ідентифікаційний номер', 'Unified_Status']].copy()
crm_combined = pd.concat([micro_cut, small_cut])

# Пріоритет: Success > In Process > Failed > Unknown > NaN
# Математично ми шукаємо max(P(status)) для кожного ID
status_priority = {'Success': 4, 'In Process': 3, 'Failed': 2, 'Unknown': 1}
crm_combined['Priority'] = crm_combined['Unified_Status'].map(status_priority)

# Сортуємо та залишаємо запис із найвищим пріоритетом (найкращий статус по клієнту)
crm_combined = crm_combined.sort_values('Priority', ascending=False).drop_duplicates('Ідентифікаційний номер')
crm_combined = crm_combined.rename(columns={'Ідентифікаційний номер': 'IDENTIFYCODE'})

# ==========================================
# 4. Злиття з базою Pilot
# ==========================================
# Left join, оскільки Pilot - це наша базова множина P
master_df = pilot_df.merge(crm_combined[['IDENTIFYCODE', 'Unified_Status']], on='IDENTIFYCODE', how='left')

# Розмічаємо булеві прапорці для агрегації
master_df['Is_Provided'] = 1
# Вважаємо, що взяли в роботу, якщо клієнт взагалі є в базі Micro або Small
master_df['Is_Worked'] = master_df['Unified_Status'].notna().astype(int) 
master_df['Is_Success'] = (master_df['Unified_Status'] == 'Success').astype(int)

# ==========================================
# 5. Розрахунок метрик та транспонування
# ==========================================
# Створюємо словник для зберігання результатів
metrics_dict = {}

for month, group in master_df.groupby('MONTH'):
    total_provided = group['Is_Provided'].sum()
    total_worked = group['Is_Worked'].sum()
    total_success = group['Is_Success'].sum()
    
    # Середня схильність для тих, кого залучили
    prop_success = group[group['Is_Success'] == 1]['PRIMARY'].mean()
    
    # Середня схильність для тих, кого НЕ залучили (з тих, кого взагалі намагались)
    # Або можна рахувати від усіх (включаючи тих, кому не дзвонили). Візьмемо відносно всіх не залучених.
    prop_failed = group[group['Is_Success'] == 0]['PRIMARY'].mean()
    
    metrics_dict[month] = {
        '1. К-сть наданих клієнтів (потенціал)': total_provided,
        '2. К-сть взятих в обробку': total_worked,
        '3. К-сть залучених (Success)': total_success,
        '4. Середня схильність залучених (PRIMARY)': f"{prop_success:.2%}" if pd.notna(prop_success) else "0.00%",
        '5. Середня схильність НЕ залучених (PRIMARY)': f"{prop_failed:.2%}" if pd.notna(prop_failed) else "0.00%",
        'Додатково: Conversion Rate (від взятих в роботу)': f"{(total_success/total_worked):.2%}" if total_worked > 0 else "0.00%"
    }

# Перетворюємо в DataFrame: колонки - місяці, рядки - метрики
report_df = pd.DataFrame(metrics_dict)

# Виводимо або зберігаємо
print(report_df)
report_df.to_excel('Model_Metrics_Report.xlsx')