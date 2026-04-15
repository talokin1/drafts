import pandas as pd
import numpy as np

# --- 1. ПІДГОТОВКА ФІЧЕЙ ДО АГРЕГАЦІЇ ---

# Створюємо флаг для позашляховиків
df_inf['is_suv'] = df_inf['category_name'].apply(lambda x: 1 if 'Позашляховик' in str(x) else 0)

# ПРИМУСОВЕ ПЕРЕТВОРЕННЯ В ЧИСЛА (захист від TypeError: agg function failed)
# Ці колонки обов'язково мають бути числовими для підрахунку mean() та sum()
cols_to_numeric = ['price_usd', 'exchange_possible', 'has_report']

for col in cols_to_numeric:
    if col in df_inf.columns:
        # Якщо в колонці затесався текст 'Missing', заміняємо його на 0
        if df_inf[col].dtype == 'object':
            # Додаємо мапінг для логічних значень, якщо вони збережені як текст
            mapping = {'True': 1, 'False': 0, 'Так': 1, 'Ні': 0, 'Missing': 0}
            df_inf[col] = df_inf[col].replace(mapping)
        
        # Перетворюємо в float, будь-яке сміття стане NaN, яке ми одразу заповнимо нулями
        df_inf[col] = pd.to_numeric(df_inf[col], errors='coerce').fillna(0)


# --- 2. АГРЕГАЦІЯ НА РІВНІ КЛІЄНТА ---

client_results = df_inf.groupby('MOBILEPHONE').agg(
    cars_count=('MOBILEPHONE', 'count'),
    max_hnwi_prob=('hnwi_prob', 'max'),
    is_potential_hnwi=('is_hnwi_car', 'max'),
    
    # Фінансові метрики
    avg_price=('price_usd', 'mean'),
    max_price=('price_usd', 'max'),
    total_garage_value=('price_usd', 'sum'), # Загальна вартість "гаража"
    
    # Поведінкові патерни
    exchange_affinity=('exchange_possible', 'mean'), # Чи схильний клієнт до обміну?
    has_report_rate=('has_report', 'mean'),          # Чи цікавлять його перевірені авто?
    
    # Категорійні флаги
    has_luxury=('mark_group', lambda x: 1 if 'Luxury' in x.values else 0),
    has_suv=('is_suv', 'max')
).reset_index()

# Заповнюємо можливі NaN нулями
client_results.fillna(0, inplace=True)


# --- 3. ПОБУДОВА БІЗНЕС-ПОРТРЕТУ (LIFT АНАЛІЗ) ---

# Список фічей для порівняння (без лізингу)
features_to_profile = [
    'cars_count',
    'avg_price', 
    'max_price', 
    'total_garage_value',
    'exchange_affinity', 
    'has_report_rate',
    'has_luxury', 
    'has_suv'
]

portrait = []
is_hnwi_mask = client_results['is_potential_hnwi'] == 1

for feat in features_to_profile:
    mean_all = client_results[feat].mean()
    mean_hnwi = client_results[is_hnwi_mask][feat].mean()
    
    # Щоб не було ділення на 0, додаємо безпеку
    lift = (mean_hnwi / mean_all) if mean_all > 0 else 0
    
    portrait.append({
        'Ознака': feat,
        'Середнє (Всі)': mean_all,
        'Середнє (HNWI)': mean_hnwi,
        'Lift': lift
    })

portrait_df_numeric = pd.DataFrame(portrait)


# --- 4. МАПІНГ НАЗВ ТА ВИВІД РЕЗУЛЬТАТІВ ---

feature_names_mapping = {
    'cars_count': 'Кількість автомобілів',
    'avg_price': 'Середня ціна авто ($)',
    'max_price': 'Максимальна ціна авто ($)',
    'total_garage_value': 'Сумарна вартість гаража ($)',
    'exchange_affinity': 'Схильність до Trade-In (0-1)',
    'has_report_rate': 'Частка авто з перевіркою (0-1)',
    'has_luxury': 'Наявність Luxury авто (Флаг)',
    'has_suv': 'Наявність SUV/Кросовера (Флаг)'
}

# Замінюємо технічні назви на людські
portrait_df_numeric['Ознака'] = portrait_df_numeric['Ознака'].map(feature_names_mapping).fillna(portrait_df_numeric['Ознака'])

# Вивід у Jupyter з красивим форматуванням
display(portrait_df_numeric.sort_values(by='Lift', ascending=False).style.format({
    'Середнє (Всі)': "{:.2f}",
    'Середнє (HNWI)': "{:.2f}",
    'Lift': "{:.2f}x"
}))