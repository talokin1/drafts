# 1. Визначаємо список нових колонок
new_kved_cols = [
    'KVED', 'KVED_DESCR',
    'KVED_2', 'KVED_2_DESCR',
    'KVED_3', 'KVED_3_DESCR',
    'KVED_4', 'KVED_4_DESCR',
    'KVED_5', 'KVED_5_DESCR'
]

# 2. Знаходимо індекс, де починаються старі колонки (щоб знати, куди вставляти)
# Якщо FIRM_KVED немає, код спробує знайти FIRM_KVEDNM, або вставить в кінець
if 'FIRM_KVED' in fin_ind.columns:
    insert_loc = fin_ind.columns.get_loc('FIRM_KVED')
elif 'FIRM_KVEDNM' in fin_ind.columns:
    insert_loc = fin_ind.columns.get_loc('FIRM_KVEDNM')
else:
    insert_loc = len(fin_ind.columns)

# 3. Виконуємо злиття (merge)
# Додаємо нові дані, вони поки що опиняться в кінці
fin_ind = fin_ind.merge(ubki[['IDENTIFYCODE'] + new_kved_cols], on='IDENTIFYCODE', how='left')

# 4. Формуємо новий порядок колонок
# Беремо всі колонки, що зараз є в датафреймі
cols = list(fin_ind.columns)

# Видаляємо старі колонки зі списку та нові (бо ми їх вставимо вручну)
cols_to_remove = ['FIRM_KVED', 'FIRM_KVEDNM'] + new_kved_cols
base_cols = [c for c in cols if c not in cols_to_remove]

# Вставляємо нові колонки у збережену позицію insert_loc
# (враховуємо зсув, якщо insert_loc був далі, ніж видалені колонки, але зазвичай це працює коректно для заміни)
final_cols = base_cols[:insert_loc] + new_kved_cols + base_cols[insert_loc:]

# 5. Застосовуємо порядок
fin_ind = fin_ind[final_cols]