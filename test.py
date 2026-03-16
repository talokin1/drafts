# 1. Формуємо Dim_Clients (беремо REGISTERDATE)
df_dim_clients = pd.merge(
    df_potential,
    df_engaged[['IDENTIFYCODE', 'CONTRAGENTID', 'REGISTERDATE']], 
    on='IDENTIFYCODE',
    how='left'
)

# Перейменовуємо REGISTERDATE на pilot_month
df_dim_clients.rename(columns={'REGISTERDATE': 'pilot_month'}, inplace=True)

# 2. Витягуємо ТІЛЬКИ картки з таблиці доходів (без pilot_month, бо він вже є)
df_static_info = df_income[['CONTRAGENTID', 'ZKP_NB_CARDS_2026_02']].copy()
df_static_info.rename(columns={'ZKP_NB_CARDS_2026_02': 'Cards_Count'}, inplace=True)
df_static_info = df_static_info.drop_duplicates(subset=['CONTRAGENTID'])

# 3. Підтягуємо картки
df_dim_clients = pd.merge(df_dim_clients, df_static_info, on='CONTRAGENTID', how='left')

# 4. Проставляємо статуси
df_dim_clients['Is_Engaged'] = df_dim_clients['CONTRAGENTID'].notna().astype(int)
df_dim_clients['Status'] = np.where(df_dim_clients['Is_Engaged'] == 1, 'Engaged', 'Potential')