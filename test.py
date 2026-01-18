# 1. Функція очищення: видаляє коми та пробіли, робить float
def clean_currency(x):
    if isinstance(x, str):
        # Видаляємо коми і пробіли
        clean_str = x.replace(',', '').replace(' ', '')
        try:
            return float(clean_str)
        except ValueError:
            return np.nan # Якщо там справді текст, ставимо NaN
    return x

# 2. Застосовуємо до всіх колонок, які зараз мають тип Object (крім реальних категорій)
# Спочатку визначимо, які колонки є типу object
obj_cols = df.select_dtypes(include=['object']).columns

print(f"Очищуємо {len(obj_cols)} колонок від сміття...")

for col in obj_cols:
    # Пропускаємо явні категорії, якщо ви їх вже задали (наприклад КВЕД)
    # Але якщо КВЕД виглядає як "50.2" (текст), це ок. 
    # Головне - не чіпати колонки, які МАЮТЬ бути текстом (якщо такі є, окрім FIRM_SIZE)
    
    # Спробуємо конвертувати
    # Евристика: якщо в колонці є коми або цифри - чистимо
    df[col] = df[col].apply(clean_currency)

# 3. Повторна конвертація типів (важливий крок!)
# Те, що стало числами - у float. Решта (якщо лишилась) - у category.
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].astype('float32')
        except:
            print(f"Колонка {col} залишилась категоріальною")
            df[col] = df[col].astype('category')

# 4. ОБРОБКА ТАРГЕТУ (Log Warning Fix)
# Рахунки можуть бути від'ємними (овердрафт). log(-100) = NaN.
# Варіант А (простий): Обрізати мінімум нулем
df['CURR_ACC'] = df['CURR_ACC'].clip(lower=0) 

# Тепер ваш старий код:
y = df['CURR_ACC']
y_log = np.log1p(y) 

# Перевірка на NaN перед сплітом (критично!)
print(f"NaN у таргеті: {y_log.isna().sum()}")
if y_log.isna().sum() > 0:
    # Видаляємо рядки з NaN
    mask = ~y_log.isna()
    X = X[mask]
    y_log = y_log[mask]