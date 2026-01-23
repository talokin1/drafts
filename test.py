# Список нових колонок, які ми додаємо з ubki
new_kved_cols = [
    'KVED', 'KVED_DESCR',
    'KVED_2', 'KVED_2_DESCR',
    'KVED_3', 'KVED_3_DESCR',
    'KVED_4', 'KVED_4_DESCR',
    'KVED_5', 'KVED_5_DESCR'
]

# 1. Знаходимо місце вставки (індекс старої колонки FIRM_KVED)
# Це потрібно зробити ДО мерджу та видалення
if 'FIRM_KVED' in fin_ind.columns:
    insert_loc = fin_ind.columns.get_loc('FIRM_KVED')
elif 'FIRM_KVEDNM' in fin_ind.columns:
    insert_loc = fin_ind.columns.get_loc('FIRM_KVEDNM')
else:
    insert_loc = len(fin_ind.columns)

# Зберігаємо список "базових" колонок (все крім тих, що будемо замінювати)
cols_to_remove = ['FIRM_KVED', 'FIRM_KVEDNM']
base_cols = [c for c in fin_ind.columns if c not in cols_to_remove]

# 2. Виконуємо злиття (Left Join)
# fin_ind залишається головним, підтягуємо дані з ubki
fin_ind = fin_ind.merge(ubki[['IDENTIFYCODE'] + new_kved_cols], on='IDENTIFYCODE', how='left')

# 3. КЛЮЧОВИЙ МОМЕНТ: Зберігаємо старі дані для рядків, де не знайшлося відповідності в ubki
# Якщо в новій колонці KVED пусто (NaN), беремо значення зі старої FIRM_KVED
if 'FIRM_KVED' in fin_ind.columns:
    fin_ind['KVED'] = fin_ind['KVED'].fillna(fin_ind['FIRM_KVED'])

# Аналогічно для опису: якщо KVED_DESCR пусто, беремо з FIRM_KVEDNM
if 'FIRM_KVEDNM' in fin_ind.columns:
    fin_ind['KVED_DESCR'] = fin_ind['KVED_DESCR'].fillna(fin_ind['FIRM_KVEDNM'])

# 4. Формуємо фінальний порядок колонок
# Розрізаємо список базових колонок у точці insert_loc і вставляємо туди блок нових колонок
# Оскільки ми використовуємо base_cols (без старих FIRM_...), старі колонки автоматично зникнуть з фінального датафрейму
final_cols = base_cols[:insert_loc] + new_kved_cols + base_cols[insert_loc:]

fin_ind = fin_ind[final_cols]