import pandas as pd
import numpy as np

# ==========================================
# ЕТАП 1: Препроцесинг та Агрегація (Identical to Train)
# ==========================================
print("1. Початок препроцесингу...")
df_inf = full_df.copy()

# 1. Синтетичні фічі (обов'язково додаємо ті самі, що і в трейні!)
df_inf['leasing_to_price_ratio'] = df_inf['min_month_leasing_pay'] / df_inf['price_usd']
df_inf['is_luxury_car'] = (df_inf['mark_group'] == 'Luxury').astype(int)

# 2. Надійний мапінг булевих ознак
bool_like_cols = ['is_abroad', 'has_report', 'auction_possible', 'exchange_possible', 
                  'is_leasing', 'verified_by_inspection', 'realty_exchange', 'technical_checked']
for col in bool_like_cols:
    if col in df_inf.columns:
        clean_series = df_inf[col].astype(str).str.strip().str.lower()
        df_inf[col] = clean_series.map({'true': 1, '1.0': 1, '1': 1, 'false': 0, '0.0': 0, '0': 0})

# 3. Числові ознаки
numeric_cols = ['doors_count', 'mileage', 'price_usd', 'min_month_leasing_pay']
for col in numeric_cols:
    if col in df_inf.columns:
        df_inf[col] = df_inf[col].replace('Missing value', np.nan)
        df_inf[col] = pd.to_numeric(df_inf[col], errors='coerce')

# 4. Агрегація
agg_funcs = {
    'price_usd': ['count', 'mean', 'max', 'sum'],
    'min_month_leasing_pay': ['mean', 'max'],
    'leasing_to_price_ratio': ['mean', 'max'],
    'mileage': ['mean', 'min'],
    'is_luxury_car': 'max', 
    'doors_count': 'max',
    'has_report': 'max',
    'is_abroad': 'max',
    'category_name': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'Missing',
    'exchange_type': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'Missing'
}

# Відфільтровуємо лише ті колонки з agg_funcs, які реально є в full_df
available_agg_funcs = {k: v for k, v in agg_funcs.items() if k in df_inf.columns}

print("2. Агрегація клієнтів...")
df_inf_client = df_inf.groupby('MOBILEPHONE').agg(available_agg_funcs)
df_inf_client.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_inf_client.columns.values]

df_inf_client = df_inf_client.rename(columns={
    'price_usd_count': 'cars_count',
    'category_name_<lambda>': 'primary_category',
    'exchange_type_<lambda>': 'primary_exchange'
})
df_inf_client.reset_index(inplace=True)

# ==========================================
# ЕТАП 2: Інференс
# ==========================================
print("3. Запуск моделі...")
X_inf = df_inf_client.drop(columns=['MOBILEPHONE'])

# Використовуємо ті самі категоріальні фічі
cat_features = X_inf.select_dtypes(include=['object', 'category']).columns.tolist()
X_inf[cat_features] = X_inf[cat_features].fillna('Missing').astype(str)

# ПРОГНОЗ. Використовуємо найкращий поріг, знайдений під час навчання
# (Підстав сюди середній best_threshold зі свого попереднього скрипта, напр. 0.85)
INFERENCE_THRESHOLD = 0.85 

df_inf_client['hnwi_probability'] = model.predict_proba(X_inf)[:, 1]
df_inf_client['is_potential_hnwi'] = (df_inf_client['hnwi_probability'] >= INFERENCE_THRESHOLD).astype(int)

# ==========================================
# ЕТАП 3: Оцінка (Sanity Checks) та Портрет
# ==========================================
total_clients = len(df_inf_client)
hnwi_count = df_inf_client['is_potential_hnwi'].sum()
hit_rate = hnwi_count / total_clients

print("\n--- SANITY CHECK METRICS ---")
print(f"Всього унікальних клієнтів: {total_clients}")
print(f"Знайдено потенційних HNWI: {hnwi_count}")
print(f"Hit Rate (частка HNWI): {hit_rate:.2%} (Адекватний рендж: 0.5% - 5.0%)")

# Перевірка на вже відомих (якщо train_dataset імпортований)
# Це дозволить перевірити Recall "в дикій природі"
try:
    known_hnwi_phones = train_dataset[train_dataset['is_hnwi'] == 1]['MOBILEPHONE'].unique()
    found_known = df_inf_client[(df_inf_client['MOBILEPHONE'].isin(known_hnwi_phones)) & (df_inf_client['is_potential_hnwi'] == 1)]
    recall_on_known = len(found_known) / len(known_hnwi_phones)
    print(f"Recall на відомих HNWI з трейну: {recall_on_known:.2%}")
except NameError:
    pass # Якщо train_dataset недоступний в цьому середовищі

print("\n--- БІЗНЕС-ПОРТРЕТ HNWI (LIFT АНАЛІЗ) ---")
# Формуємо профіль
features_to_profile = ['price_usd_sum', 'price_usd_mean', 'cars_count', 'leasing_to_price_ratio_mean', 'is_luxury_car_max']

portrait = []
for feat in features_to_profile:
    if feat in df_inf_client.columns:
        mean_all = df_inf_client[feat].mean()
        mean_hnwi = df_inf_client[df_inf_client['is_potential_hnwi'] == 1][feat].mean()
        
        # Захист від ділення на нуль
        lift = (mean_hnwi / mean_all) if mean_all > 0 else 0 
        
        portrait.append({
            'Ознака': feat,
            'Середнє (Всі)': f"{mean_all:.2f}",
            'Середнє (HNWI)': f"{mean_hnwi:.2f}",
            'Lift': f"{lift:.2f}x"
        })

portrait_df = pd.DataFrame(portrait)
print(portrait_df.to_string(index=False))

# Зберігаємо результати для передачі замовнику (Viktoria Myslyva)
final_export = df_inf_client[df_inf_client['is_potential_hnwi'] == 1].sort_values(by='hnwi_probability', ascending=False)
# final_export.to_csv('potential_hnwi_clients.csv', index=False)