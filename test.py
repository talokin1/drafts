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