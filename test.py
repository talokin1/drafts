# 1. Створюємо словник мапінгу
brand_mapping = {
    # ---------------------------------------------------------
    # 1. Luxury / Exotic (Найвища ймовірність HNWI сегменту)
    # ---------------------------------------------------------
    'porsche': 'Luxury', 'mclaren': 'Luxury', 'jaguar': 'Luxury', 
    'land-rover': 'Luxury', 'lexus': 'Luxury', 'zeekr': 'Luxury', 
    
    # ---------------------------------------------------------
    # 2. Premium (Високий середній клас та вище)
    # ---------------------------------------------------------
    'bmw': 'Premium', 'audi': 'Premium', 'mercedes-benz': 'Premium', 
    'volvo': 'Premium', 'infiniti': 'Premium', 'tesla': 'Premium', 
    'cadillac': 'Premium', 'lincoln': 'Premium', 'acura': 'Premium', 
    'alfa-romeo': 'Premium', 'chrysler': 'Premium', 'mini': 'Premium',
    
    # ---------------------------------------------------------
    # 3. Mid-Range (Мас-маркет, основа датасету)
    # ---------------------------------------------------------
    'volkswagen': 'Mid_Range', 'nissan': 'Mid_Range', 'renault': 'Mid_Range', 
    'ford': 'Mid_Range', 'opel': 'Mid_Range', 'skoda': 'Mid_Range', 
    'toyota': 'Mid_Range', 'hyundai': 'Mid_Range', 'kia': 'Mid_Range', 
    'mazda': 'Mid_Range', 'mitsubishi': 'Mid_Range', 'chevrolet': 'Mid_Range', 
    'peugeot': 'Mid_Range', 'honda': 'Mid_Range', 'subaru': 'Mid_Range', 
    'jeep': 'Mid_Range', 'citroen': 'Mid_Range', 'seat': 'Mid_Range', 
    'fiat': 'Mid_Range', 'suzuki': 'Mid_Range', 'dodge': 'Mid_Range', 
    'ssangyong': 'Mid_Range', 'buick': 'Mid_Range', 'lancia': 'Mid_Range', 
    'byd': 'Mid_Range',
    
    # ---------------------------------------------------------
    # 4. Budget (Бюджетний сегмент, авто з пробігом/китайські/радянські)
    # ---------------------------------------------------------
    'vaz-lada': 'Budget', 'daewoo': 'Budget', 'zaz': 'Budget', 'gaz': 'Budget', 
    'dacia': 'Budget', 'smart': 'Budget', 'great-wall': 'Budget', 'geely': 'Budget', 
    'uaz': 'Budget', 'izh': 'Budget', 'chery': 'Budget', 'haval': 'Budget', 
    'moskvich-azlk': 'Budget', 'kastom': 'Budget', 'samodelnyj': 'Budget', 
    'bogdan': 'Budget',
    
    # ---------------------------------------------------------
    # 5. Commercial / Moto / Special (Тягачі, причепи, спецтехніка, мото)
    # ---------------------------------------------------------
    'kogel': 'Commercial_Moto', 'daf': 'Commercial_Moto', 'kovi': 'Commercial_Moto', 
    'yamaha': 'Commercial_Moto', 'iveco': 'Commercial_Moto', 'spark': 'Commercial_Moto', 
    'schwarzmuller': 'Commercial_Moto', 'carnehl': 'Commercial_Moto', 'kayo': 'Commercial_Moto', 
    'krone': 'Commercial_Moto', 'shineray': 'Commercial_Moto', 'szap': 'Commercial_Moto', 
    'benalu': 'Commercial_Moto', 'hendricks': 'Commercial_Moto', 'floor': 'Commercial_Moto', 
    'bumar': 'Commercial_Moto', 'krukenmeier': 'Commercial_Moto', 'schmitz': 'Commercial_Moto', 
    'kawasaki': 'Commercial_Moto', 'ovibos': 'Commercial_Moto', 'menci': 'Commercial_Moto', 
    'gras': 'Commercial_Moto', 'scania': 'Commercial_Moto', 'pricep': 'Commercial_Moto', 
    'bodex': 'Commercial_Moto', 'musstang': 'Commercial_Moto', 'liebherr': 'Commercial_Moto', 
    'general-trailers': 'Commercial_Moto', 'van-hool': 'Commercial_Moto', 'wirtgen': 'Commercial_Moto', 
    'paz': 'Commercial_Moto', 'harley-davidson': 'Commercial_Moto', 'brp': 'Commercial_Moto', 
    'forte': 'Commercial_Moto', 'tata': 'Commercial_Moto', 'schmitz-cargobull': 'Commercial_Moto', 
    'jianshe': 'Commercial_Moto', 'inter-cars': 'Commercial_Moto', 'ducati': 'Commercial_Moto', 
    'burg': 'Commercial_Moto', 'pg': 'Commercial_Moto', 'bajaj': 'Commercial_Moto', 
    'hunter': 'Commercial_Moto', 'ep-equipment': 'Commercial_Moto', 'odaz': 'Commercial_Moto', 
    'ekho': 'Commercial_Moto', 'kelberg': 'Commercial_Moto', 'baz': 'Commercial_Moto', 
    'panissars': 'Commercial_Moto', 'loncin': 'Commercial_Moto', 'bailey-discovery': 'Commercial_Moto', 
    'fliegl': 'Commercial_Moto', 'case-construction': 'Commercial_Moto', 'kaya': 'Commercial_Moto', 
    'triumph': 'Commercial_Moto', 'kraz': 'Commercial_Moto', 'man': 'Commercial_Moto', 
    'kazuma': 'Commercial_Moto'
}

# 2. Застосовуємо мапінг до датафрейму
# Використовуємо .map() і на всякий випадок .fillna('Other'), 
# якщо в тестовій вибірці з'явиться марка, якої не було в трені
train_dataset['mark_group'] = train_dataset['mark_name'].map(brand_mapping).fillna('Other')

# 3. Видаляємо стару колонку, щоб вона не створювала шум
train_dataset = train_dataset.drop(columns=['mark_name'])

# Перевірка розподілу нових груп
print(train_dataset['mark_group'].value_counts(dropna=False))