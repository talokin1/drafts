# 2. Витягуємо компоненти коду з колонки KVED

# Клас — повний KVED
df['Class_Code'] = df['KVED']

# Розділ — перші 2 символи
df['Division_Code'] = df['KVED'].str[:2]

# Група — формат XX.X (краще не просто [:4], а по split)
df['Group_Code'] = df['KVED'].apply(lambda x: '.'.join(x.split('.')[:2]) if '.' in x else x)

# Section_Code уже формується через division_to_section_code
df['Section_Code'] = df['Division_Code'].map(division_to_section_code)




df['Class_Name'] = df['Class_Code'].map(classes_name_map)
df['Group_Name'] = df['Group_Code'].map(groups_name_map)
df['Division_Name'] = df['Division_Code'].map(divisions_name_map)
df['Section_Name'] = df['Section_Code'].map(sections_name_map)



final_df = df.drop(columns=['div_key', 'group_key'], errors='ignore')
