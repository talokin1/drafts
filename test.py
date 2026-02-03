# 1. KVED HIERARCHY (Розбиваємо код на рівні)
# Припускаємо, що формат КВЕД це "XX.XX" (наприклад, "01.11")
df['KVED'] = df['KVED'].astype(str)

# Перші 2 цифри (Розділ) - це широка індустрія (IT, Агро, Металургія)
df['KVED_DIV'] = df['KVED'].apply(lambda x: x.split('.')[0] if '.' in x else x[:2])

# Перші 3 символи (Група) - уточнення (наприклад 01.1 - вирощування однорічних)
df['KVED_GROUP'] = df['KVED'].apply(lambda x: x[:4] if '.' in x else x[:3])

print(f"Created KVED hierarchy features. Unique Divisions: {df['KVED_DIV'].nunique()}")

# ---------------------------------------------------------

# 2. SMOOTHED TARGET ENCODING (TE)
# Ця функція замінює категорію на середнє значення таргету з регуляризацією
# Це краще, ніж звичайне середнє, бо не дає шуму на рідкісних категоріях

def calc_smooth_mean(df, by, on, m):
    # by - колонка для групування (KVED)
    # on - таргет (CURR_ACC або його логарифм)
    # m - "вага" загального середнього (smoothing factor)
    
    # Середнє глобальне
    mean = df[on].mean()
    
    # Агрегація по категорії: сума та кількість
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    # Формула згладжування:
    # (n * category_mean + m * global_mean) / (n + m)
    # Чим менше записів (n), тим сильніше тягнемо до глобального середнього
    smooth = (counts * means + m * mean) / (counts + m)
    
    return df[by].map(smooth)

# Важливо: TE краще робити на логарифмованому таргеті, щоб викиди не ламали середнє
# Створимо тимчасову колонку з логом (якщо ще немає)
temp_target = np.log1p(df['CURR_ACC']) 

# Робимо TE для рівнів ієрархії
# m=10 або m=20 - хороша вага для згладжування
df['KVED_TE'] = calc_smooth_mean(df, by='KVED', on='CURR_ACC', m=10) # Використовуємо оригінальну шкалу або лог - спробуй лог
df['KVED_DIV_TE'] = calc_smooth_mean(df, by='KVED_DIV', on='CURR_ACC', m=10)

# Я б радив спробувати TE саме на лог-таргеті:
df['KVED_LOG_TE'] = calc_smooth_mean(df, by='KVED', on=temp_target.name if hasattr(temp_target, 'name') else 'log_target', m=10)

print("Target Encoding features created.")

# ---------------------------------------------------------
# Не забудь додати нові колонки у список фічей
# І переконайся, що 'KVED_DIV' та 'KVED_GROUP' переведені в category тип, якщо хочеш їх лишити як категоріальні
categorical_add = ['KVED_DIV', 'KVED_GROUP']
for c in categorical_add:
    df[c] = df[c].astype('category')