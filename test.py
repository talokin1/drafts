import pandas as pd

# Припускаємо, що df_inf_client вже має колонки 'hnwi_probability' та 'is_potential_hnwi'

print("\n--- 1. РОЗПОДІЛ ТА МАСШТАБ (SANITY CHECK) ---")
total_clients = len(df_inf_client)
hnwi_count = df_inf_client['is_potential_hnwi'].sum()
hit_rate = hnwi_count / total_clients

print(f"Всього унікальних клієнтів: {total_clients}")
print(f"Знайдено потенційних HNWI: {hnwi_count}")
print(f"Hit Rate (частка відібраних): {hit_rate:.2%} (Очікуємо до 5%)")

print("\n--- 2. БІЗНЕС-ПОРТРЕТ СЕГМЕНТУ (LIFT АНАЛІЗ) ---")
# Обираємо ключові фічі, які мають сенс для бізнесу
features_to_profile = [
    'price_usd_sum',            # Сумарний капітал в авто
    'price_usd_mean',           # Середня вартість одного авто
    'cars_count',               # Кількість авто
    'leasing_to_price_ratio_mean', 
    'is_luxury_car_max',        # Відсоток клієнтів з хоча б одним Luxury авто
    'doors_count_max'           # Контрольна фіча (Lift має бути ~1.0, бо двері не залежать від багатства)
]

portrait = []
for feat in features_to_profile:
    if feat in df_inf_client.columns:
        mean_all = df_inf_client[feat].mean()
        mean_hnwi = df_inf_client[df_inf_client['is_potential_hnwi'] == 1][feat].mean()
        
        # Lift показує, у скільки разів показник HNWI вищий за середній по базі
        lift = (mean_hnwi / mean_all) if mean_all > 0 else 0 
        
        portrait.append({
            'Ознака': feat,
            'Середнє (Всі)': f"{mean_all:.2f}",
            'Середнє (HNWI)': f"{mean_hnwi:.2f}",
            'Lift': f"{lift:.2f}x"
        })

portrait_df = pd.DataFrame(portrait)
print(portrait_df.to_string(index=False))

print("\n--- 3. ТОП-5 НАЙБІЛЬШ ЙМОВІРНИХ КЛІЄНТІВ (MANUAL REVIEW) ---")
# Виводимо 5 людей з найвищою ймовірністю, щоб оцінити їх "очима"
top_5_examples = df_inf_client.sort_values(by='hnwi_probability', ascending=False).head(5)
cols_to_show = ['MOBILEPHONE', 'hnwi_probability', 'price_usd_sum', 'cars_count', 'is_luxury_car_max', 'primary_category']
print(top_5_examples[cols_to_show].to_string(index=False))

# Збереження повної бази лідів для керівництва
top_candidates = df_inf_client[df_inf_client['is_potential_hnwi'] == 1].sort_values(by='hnwi_probability', ascending=False)
# top_candidates.to_csv('potential_hnwi_leads.csv', index=False)