# 1. Збираємо всі пари Клієнт-Банк
df_p_list = []
for file in all_files:
    df_temp = pd.read_excel(file)
    bank_name = os.path.basename(file).replace('_clients.xlsx', '').replace('.xlsx', '')
    temp_subset = df_temp[['IDENTIFYCODE']].copy()
    temp_subset['Source_Bank'] = bank_name
    df_p_list.append(temp_subset)

df_mapping = pd.concat(df_p_list, ignore_index=True).drop_duplicates()

# 2. Fact_Income тепер буде містити Source_Bank
# Це дозволить Power BI рахувати гроші по кожному банку окремо
df_facts_with_banks = pd.merge(df_facts, df_mapping, on='IDENTIFYCODE', how='left')

# 3. Зберігаємо
df_facts_with_banks.to_csv('Fact_Income.csv', index=False)