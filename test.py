import pandas as pd
import numpy as np

# Припускаємо, що validation_results, df та cat_cols вже існують у твоєму пайплайні
# validation_results має мати колонки: 'IDENTIFYCODE', 'Predicted', 'True_Value'
# df - це оригінальний датасет з категоріальними ознаками (KVED, Industry тощо)

# ==========================================
# 1. ПІДГОТОВКА ДАНИХ ТА БАКЕТІВ
# ==========================================

# Об'єднуємо результати з оригінальними категоріями для аналізу
# Важливо: validation_results має індекс або колонку для джойну
report_df = validation_results.merge(
    df[cat_cols + ['LIABILITIES']],  # Додаємо категорії + пасиви
    left_on='IDENTIFYCODE', 
    right_index=True, 
    how='left'
)

report_df['Difs'] = (report_df['True_Value'] - report_df['Predicted']).abs()

# --- Створення хитрих бакетів ---
# 1. Початковий хвіст (до 1к)
bins = [-1, 1000]
# 2. Основне тіло (1к - 100к з кроком 5к)
# np.arange(start, stop, step) - stop не включається, тому беремо 100001
bins.extend(np.arange(1000, 100001, 5000))
# 3. Великі клієнти (від 100к)
bins.extend([1000000, np.inf]) # Можна додати проміжні, якщо треба

# Створення лейблів (опціонально, або автоматично)
# Для автоматичних лейблів краще не передавати labels=... в cut, 
# але для краси звіту згенеруємо їх:
labels = []
for i in range(len(bins)-1):
    low = bins[i]
    high = bins[i+1]
    if high == np.inf:
        labels.append(f"{int(low/1000)}k+")
    elif low == -1:
        labels.append("0-1k")
    else:
        labels.append(f"{int(low/1000)}k-{int(high/1000)}k")

report_df['Income_Range'] = pd.cut(report_df['Predicted'], bins=bins, labels=labels)

# ==========================================
# 2. ЗВЕДЕНА ТАБЛИЦЯ (ANALYTICS)
# ==========================================
summary_table = report_df.groupby('Income_Range', observed=False)['Difs'].agg(
    Count='count',
    Mean_Error='mean',
    Median_Error='median',
    Avg_Predicted=('Predicted', 'mean') # Середній прогноз у бакеті
).reset_index()

# ==========================================
# 3. FEATURE VALUE DRIVERS (Які значення найкорисніші)
# ==========================================
# Ми дивимось, які категорії мають найвищий середній Predicted Income.
# Це відповідає на питання "Які значення приносять користь".

drivers_list = []

for col in cat_cols:
    # Групуємо по значенню категорії
    cat_stats = report_df.groupby(col).agg(
        Count=('IDENTIFYCODE', 'count'),
        Avg_Pred=('Predicted', 'mean'),
        Avg_Fact=('True_Value', 'mean')
    ).reset_index()
    
    # Фільтруємо рідкісні категорії (шум), наприклад, менше 10 записів
    cat_stats = cat_stats[cat_stats['Count'] > 10].copy()
    
    # Сортуємо: що модель оцінює найдорожче (Top Potential)
    cat_stats = cat_stats.sort_values(by='Avg_Pred', ascending=False).head(10)
    
    # Додаємо мета-дані
    cat_stats['Feature_Name'] = col
    cat_stats = cat_stats.rename(columns={col: 'Value_Name'})
    
    # Перевпорядковуємо колонки
    cat_stats = cat_stats[['Feature_Name', 'Value_Name', 'Count', 'Avg_Pred', 'Avg_Fact']]
    drivers_list.append(cat_stats)

# Збираємо все в одну таблицю
if drivers_list:
    drivers_table = pd.concat(drivers_list, ignore_index=True)
else:
    drivers_table = pd.DataFrame(columns=['Feature_Name', 'Value_Name', 'Count', 'Avg_Pred', 'Avg_Fact'])

# ==========================================
# 4. ЕКСПОРТ В EXCEL
# ==========================================
file_name = 'Model_Analysis_Report_v3.xlsx'

with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
    workbook = writer.book
    
    # --- ФОРМАТИ ---
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
    num_fmt = workbook.add_format({'num_format': '#,##0.00'})
    
    # === ЛИСТ 1: Report (Аналіз помилок) ===
    sheet_name_1 = 'Validation_Report'
    # Основна таблиця (перші 100 рядків для прикладу або повна)
    report_df.head(100).to_excel(writer, sheet_name=sheet_name_1, startrow=1, index=False)
    
    # Зведена таблиця справа (startcol=10 для відступу)
    summary_table.to_excel(writer, sheet_name=sheet_name_1, startrow=1, startcol=10, index=False)
    
    worksheet1 = writer.sheets[sheet_name_1]
    
    # Заголовки
    worksheet1.write(0, 0, "Detail Report (Top 100)", header_fmt)
    worksheet1.write(0, 10, "Binned Error Analysis", header_fmt)
    
    # Ширина колонок
    worksheet1.set_column('A:I', 12, num_fmt)
    worksheet1.set_column('K:K', 20) # Range Name
    worksheet1.set_column('L:N', 15, num_fmt)

    # === ЛИСТ 2: Drivers (Аналіз значень фічей) ===
    sheet_name_2 = 'Feature_Drivers'
    drivers_table.to_excel(writer, sheet_name=sheet_name_2, startrow=1, index=False)
    
    worksheet2 = writer.sheets[sheet_name_2]
    
    # Заголовки стовпців (перезаписуємо з форматом)
    for col_num, value in enumerate(drivers_table.columns):
        worksheet2.write(1, col_num, value, header_fmt)
        
    worksheet2.write(0, 0, "TOP Values by High Potential (Avg Prediction)", header_fmt)
    
    # Ширина та формати
    worksheet2.set_column('A:A', 20) # Feature Name
    worksheet2.set_column('B:B', 30) # Value Name (довгі назви індустрій)
    worksheet2.set_column('D:E', 15, num_fmt) # Гроші

    print(f"Report saved to {file_name}")
    print("\nTop 5 Drivers found:")
    print(drivers_table.head())