# рахуємо кількість по регіонах
region_counts = df['FIRM_REGION'].value_counts()

# переводимо у відсотки
region_percent = df['FIRM_REGION'].value_counts(normalize=True) * 100

# беремо тільки потрібні регіони
target_regions = ['Харківська обл.', 'Полтавська обл.', 'Сумська обл.']

result = region_percent[region_percent.index.isin(target_regions)]

print(result.round(2))