import pandas as pd

# 1. Створюємо словники для перейменування
map_q1 = {
    'Доступний аукціон': 'auction_possible',
    'Доступний обмін': 'exchange_possible',
    'Обмін на нерухомість': 'realty_exchange',
    'Тип обміну': 'exchange_type',
    'ціна (USD)': 'price_usd',
    'min_month_leasing_bu_pay_UAH': 'min_month_leasing_pay',
    'Марка авто': 'mark_name',
    'Категорія авто': 'category_name', 
    'Наявність звіту по авто': 'has_report',
    'Пройдено технічну перевірку': 'technical_checked',
    'Перевірено інспекцією': 'verified_by_inspection',
    'За кордоном': 'is_abroad',
    'Авто в кредиті': 'is_leasing',
    'К-сть дверей': 'doors_count',
    'Пробіг (тис, км)': 'mileage'
}

map_q4 = {
    'auctionPossible': 'auction_possible',
    'exchangePossible': 'exchange_possible',
    'realtyExchange': 'realty_exchange',
    'exchangeType': 'exchange_type',
    'USD': 'price_usd',
    'minMonthLeasingBuPay': 'min_month_leasing_pay',
    'markNameEng': 'mark_name',
    'subCategoryName': 'category_name',
    'haveInfotechReport': 'has_report',
    'technicalChecked': 'technical_checked',
    'verifiedByInspectionCenter': 'verified_by_inspection',
    'abroad': 'is_abroad',
    'isLeasing': 'is_leasing',
    'numberOfDoors': 'doors_count',
    'mileage': 'mileage'
}

# 2. Перейменовуємо колонки в обох датасетах
# Ті колонки, яких немає в словнику, залишаться зі старими назвами
df1 = autoria_q1.rename(columns=map_q1)
df2 = autor4q.rename(columns=map_q4)

# 3. Знаходимо спільні колонки (перетин множин)
common_cols = list(set(df1.columns) & set(df2.columns))

# 4. Фільтруємо датасети та об'єднуємо (UNION ALL)
df1_filtered = df1[common_cols]
df2_filtered = df2[common_cols]

# ignore_index=True обов'язковий, щоб індекси не дублювалися після злиття
train_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)

print(f"Розмір фінальної тренувальної вибірки: {train_df.shape}")