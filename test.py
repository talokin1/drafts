# 1. Примусова очистка від "зомбі"-колонок з минулих запусків
cols_to_drop = [c for c in X_train.columns if c.endswith('_TE')]
if cols_to_drop:
    print(f"Dropping old leaking columns: {cols_to_drop}")
    X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
    X_val = X_val.drop(columns=cols_to_drop, errors='ignore')

# 2. Додаємо таргет тимчасово
X_train['target_temp'] = y_train_log

# 3. Запускаємо ПРАВИЛЬНИЙ Target Encoding
X_train, X_val = apply_target_encoding(X_train, X_val, 'KVED_DIV', X_train['target_temp'])

# 4. Прибираємо тимчасовий таргет
X_train.drop(columns=['target_temp'], inplace=True)

# ... далі йде навчання моделі (reg = lgb.LGBMRegressor...)