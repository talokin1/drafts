# 1. Визначаємо, які з нових колонок вже є в таблиці, і видаляємо їх, 
# щоб merge не створив дублікатів типу KVED_x / KVED_y
cols_overlap = [c for c in new_kved_cols if c in fin_ind.columns]
fin_ind = fin_ind.drop(columns=cols_overlap)

# 2. Тепер ваш код спрацює коректно, бо конфлікту імен не буде
fin_ind = fin_ind.merge(ubki[['IDENTIFYCODE'] + new_kved_cols], on='IDENTIFYCODE', how='left')