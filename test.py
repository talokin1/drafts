# ==========================================
# 1. СТВОРЕННЯ ФІЧ (До спліта)
# ==========================================
# Це безпечні операції, робимо їх відразу на всьому df
# Переконайся, що ці рядки розкоментовані!
df['KVED'] = df['KVED'].astype(str)
df['KVED_DIV'] = df['KVED'].apply(lambda x: x.split('.')[0] if '.' in x else x[:2])
df['KVED_GROUP'] = df['KVED'].apply(lambda x: x[:4] if '.' in x else x[:3])

# Конвертуємо в категорії
for c in ['KVED_DIV', 'KVED_GROUP']:
    df[c] = df[c].astype('category')

# ==========================================
# 2. SPLIT
# ==========================================
# Визначаємо таргет
TARGET_COL = "CURR_ACC"
y_log = df[TARGET_COL].astype(float).copy() # (або log1p, якщо ти вже прологарифмував раніше)
X = df.drop(columns=[TARGET_COL]).copy()

# Спліт
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# ==========================================
# 3. TARGET ENCODING (Після спліта)
# ==========================================

# Функція для TE (вона сама обробить і train, і val)
def apply_target_encoding(train_df, val_df, col, target, m=10):
    # Рахуємо середні на TRAIN
    global_mean = target.mean()
    agg = train_df.groupby(col)[target.name].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + m * global_mean) / (counts + m)
    
    # Застосовуємо до TRAIN
    train_df[col + '_TE'] = train_df[col].map(smooth)
    # Застосовуємо до VAL (з заповненням пропусків для нових категорій)
    val_df[col + '_TE'] = val_df[col].map(smooth).fillna(global_mean)
    
    return train_df, val_df

# --- ВИКЛИК ФУНКЦІЇ ---

# 1. Чистимо від старих спроб (щоб не було TypeError)
cols_to_drop = [c for c in X_train.columns if c.endswith('_TE')]
if cols_to_drop:
    X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
    X_val = X_val.drop(columns=cols_to_drop, errors='ignore')

# 2. Тимчасово додаємо таргет в X_train (щоб функція його бачила)
X_train['target_temp'] = y_train_log

# 3. Робимо TE для KVED_DIV
X_train, X_val = apply_target_encoding(X_train, X_val, 'KVED_DIV', X_train['target_temp'])

# 4. Прибираємо тимчасовий таргет
X_train.drop(columns=['target_temp'], inplace=True)

print("Дані підготовлено коректно.")