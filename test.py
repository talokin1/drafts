import pandas as pd
import numpy as np

# Припускаємо, що df_inf та INFERENCE_THRESHOLD у тебе вже є

# Додамо флаг для SUV/Кросоверів, бо заможні клієнти часто їх купують
df_inf['is_suv'] = df_inf['category_name'].apply(lambda x: 1 if 'Позашляховик' in str(x) else 0)

# Розширена агрегація
client_results = df_inf.groupby('MOBILEPHONE').agg(
    cars_count=('MOBILEPHONE', 'count'),
    max_hnwi_prob=('hnwi_prob', 'max'),
    is_potential_hnwi=('is_hnwi_car', 'max'),
    
    # Грошові метрики
    avg_price=('price_usd', 'mean'),
    max_price=('price_usd', 'max'),
    total_garage_value=('price_usd', 'sum'), # Загальна вартість "гаража"
    
    # Лізинг / Кредит
    avg_leasing_pay=('min_month_leasing_pay', 'mean'),
    max_leasing_pay=('min_month_leasing_pay', 'max'),
    
    # Поведінкові патерни (частка авто з певними властивостями)
    exchange_affinity=('exchange_possible', 'mean'), # Чи схильний клієнт до обміну?
    has_report_rate=('has_report', 'mean'),          # Чи цікавлять його перевірені авто?
    
    # Категорійні флаги (чи є хоча б одне авто такого типу)
    has_luxury=('mark_group', lambda x: 1 if 'Luxury' in x.values else 0),
    has_suv=('is_suv', 'max')
).reset_index()

# Заповнюємо можливі NaN нулями (якщо лізингу немає тощо)
client_results.fillna(0, inplace=True)



features_to_profile = [
    'cars_count', 
    'avg_price', 'max_price', 'total_garage_value',
    'avg_leasing_pay', 'max_leasing_pay',
    'exchange_affinity', 'has_report_rate',
    'has_luxury', 'has_suv'
]

portrait = []
is_hnwi_mask = client_results['is_potential_hnwi'] == 1

for feat in features_to_profile:
    mean_all = client_results[feat].mean()
    mean_hnwi = client_results[is_hnwi_mask][feat].mean()
    lift = (mean_hnwi / mean_all) if mean_all > 0 else 0
    
    # Щоб не було ділення на 0, додамо безпеку
    portrait.append({
        'Ознака': feat,
        'Середнє (Всі)': mean_all,
        'Середнє (HNWI)': mean_hnwi,
        'Lift': lift
    })

portrait_df_numeric = pd.DataFrame(portrait)

# Мапінг для людських назв фічей у звіті
feature_names_mapping = {
    'cars_count': 'Кількість автомобілів',
    'avg_price': 'Середня ціна авто ($)',
    'max_price': 'Максимальна ціна авто ($)',
    'total_garage_value': 'Сумарна вартість гаража ($)',
    'avg_leasing_pay': 'Середній лізинговий платіж',
    'max_leasing_pay': 'Макс. лізинговий платіж',
    'exchange_affinity': 'Схильність до Trade-In (0-1)',
    'has_report_rate': 'Частка авто з перевіркою (0-1)',
    'has_luxury': 'Наявність Luxury авто (Флаг)',
    'has_suv': 'Наявність SUV/Кросовера (Флаг)'
}
portrait_df_numeric['Ознака'] = portrait_df_numeric['Ознака'].map(feature_names_mapping).fillna(portrait_df_numeric['Ознака'])

# Вивід для тебе в Jupyter (щоб перевірити, що все ок)
display(portrait_df_numeric.sort_values(by='Lift', ascending=False).style.format({
    'Середнє (Всі)': "{:.2f}",
    'Середнє (HNWI)': "{:.2f}",
    'Lift': "{:.2f}x"
}))