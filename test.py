import pandas as pd
import numpy as np

print("--- 1. ПІДГОТОВКА ТА ІНФЕРЕНС ---")

df_inf = full_df.copy()

# 1. Відбираємо ті самі колонки, які бачила модель під час навчання.
# model.feature_names_ гарантує 100% збіг порядку та назв колонок з X_train
expected_features = model.feature_names_
X_inf = df_inf[expected_features].copy()

# 2. Препроцесинг категоріальних фічей (ідентично до трейну)
cat_features = X_inf.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
X_inf[cat_features] = X_inf[cat_features].fillna('Missing').astype(str)

# 3. Скоринг
# Беремо середній поріг з твоєї крос-валідації (з фото ~0.494)
INFERENCE_THRESHOLD = 0.494

df_inf['hnwi_prob'] = model.predict_proba(X_inf)[:, 1]
df_inf['is_hnwi_car'] = (df_inf['hnwi_prob'] >= INFERENCE_THRESHOLD).astype(int)

print(f"Проскороно автомобілів: {len(df_inf)}")
print(f"Знайдено HNWI-автомобілів: {df_inf['is_hnwi_car'].sum()}")


print("\n--- 2. АГРЕГАЦІЯ РЕЗУЛЬТАТІВ ДО КЛІЄНТА ---")
# Бізнесу потрібні клієнти. Згрупуємо результати за номером телефону.
client_results = df_inf.groupby('MOBILEPHONE').agg(
    cars_count=('MOBILEPHONE', 'count'),
    max_hnwi_prob=('hnwi_prob', 'max'),       # Максимальна ймовірність серед усіх авто клієнта
    is_potential_hnwi=('is_hnwi_car', 'max'), # 1, якщо хоча б одне авто отримало прапорець
    avg_price=('price_usd', 'mean'),
    max_price=('price_usd', 'max'),
    has_luxury=('mark_group', lambda x: 1 if 'Luxury' in x.values else 0)
).reset_index()

total_clients = len(client_results)
hnwi_clients_count = client_results['is_potential_hnwi'].sum()
hit_rate = hnwi_clients_count / total_clients

print(f"Всього унікальних клієнтів: {total_clients}")
print(f"Знайдено потенційних HNWI-клієнтів: {hnwi_clients_count}")
print(f"Hit Rate (частка відібраних клієнтів): {hit_rate:.2%}")


print("\n--- 3. БІЗНЕС-ПОРТРЕТ HNWI (LIFT АНАЛІЗ) ---")
# Порівнюємо середні показники всіх клієнтів з відібраними HNWI
features_to_profile = ['cars_count', 'avg_price', 'max_price', 'has_luxury']

portrait = []
for feat in features_to_profile:
    mean_all = client_results[feat].mean()
    mean_hnwi = client_results[client_results['is_potential_hnwi'] == 1][feat].mean()
    
    # Lift = у скільки разів показник HNWI вищий за середній по базі
    lift = (mean_hnwi / mean_all) if mean_all > 0 else 0 
    
    portrait.append({
        'Ознака': feat,
        'Середнє (Всі)': f"{mean_all:.2f}",
        'Середнє (HNWI)': f"{mean_hnwi:.2f}",
        'Lift': f"{lift:.2f}x"
    })

portrait_df = pd.DataFrame(portrait)
print(portrait_df.to_string(index=False))


# 4. Збереження списку лідів для стейкхолдерів
top_leads = client_results[client_results['is_potential_hnwi'] == 1].sort_values(by='max_hnwi_prob', ascending=False)
# top_leads.to_csv('hnwi_leads_for_business.csv', index=False)