# 1. Виконуємо злиття (merge)
# Використовуємо 'left', щоб зберегти всі рядки з основного датафрейму fin_ind
# Також явно вказуємо колонки з ubki, які нам потрібні, плюс ключ для злиття
cols_to_merge = ['IDENTIFYCODE', 'SECTION_CODE', 'SECTION_NAME', 'DIVISION_NAME', 'GROUP_NAME']

fin_ind = fin_ind.merge(ubki[cols_to_merge], on='IDENTIFYCODE', how='left')

# 2. Перевпорядковуємо колонки, щоб нові дані були на початку
# Список нових колонок у бажаному порядку
new_cols_order = ['SECTION_CODE', 'SECTION_NAME', 'DIVISION_NAME', 'GROUP_NAME']

# Решта колонок, які вже були в fin_ind (виключаючи ті, що ми щойно додали в список вище)
remaining_cols = [col for col in fin_ind.columns if col not in new_cols_order]

# Фінальне перевпорядкування
fin_ind = fin_ind[new_cols_order + remaining_cols]