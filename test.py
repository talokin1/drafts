# 1. Спочатку робиш спліт
X_train, X_val, y_train_log, y_val_log = train_test_split(..., random_state=42)

# 2. Функція для TE, яка навчається ТІЛЬКИ на X_train
def apply_target_encoding(train_df, val_df, col, target, m=10):
    # Рахуємо середні тільки по TRAIN
    global_mean = target.mean()
    agg = train_df.groupby(col)[target.name].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + m * global_mean) / (counts + m)
    
    # Застосовуємо мапінг до TRAIN і до VAL
    train_df[col + '_TE'] = train_df[col].map(smooth)
    val_df[col + '_TE'] = val_df[col].map(smooth).fillna(global_mean) # fillna для нових категорій
    
    return train_df, val_df

# 3. Застосовуємо (для прикладу на KVED_DIV)
# Важливо: y_train_log має бути Series з індексами як у X_train
X_train['target_temp'] = y_train_log
X_train, X_val = apply_target_encoding(X_train, X_val, 'KVED_DIV', X_train['target_temp'])
X_train.drop(columns=['target_temp'], inplace=True)