# 1. Читаємо всі файли та створюємо список пар ID - Банк
df_p_list = []
for file in all_files:
    df_temp = pd.read_excel(file)
    bank_name = os.path.basename(file).replace('_clients.xlsx', '').replace('.xlsx', '')
    
    # Вибираємо тільки потрібну колонку
    temp_subset = df_temp[['IDENTIFYCODE']].copy()
    temp_subset['Source_Bank'] = bank_name
    df_p_list.append(temp_subset)

# Об'єднуємо все в одну "довгу" таблицю
df_all_potential = pd.concat(df_p_list, ignore_index=True)

# ГРУПУВАННЯ (щоб уникнути твого кейсу з дублями)
# Для кожного ID збираємо всі унікальні банки в один рядок
df_p_unique = df_all_potential.groupby('IDENTIFYCODE')['Source_Bank'].apply(
    lambda x: ', '.join(sorted(x.unique()))
).reset_index()

# Тепер у нас є унікальна множина Dim_Clients