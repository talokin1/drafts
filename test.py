# ... (твій код генерації features_to_use) ...

# 1. Явно викидаємо ID зі списку фічей, якщо він туди потрапив
id_col = 'IDENTIFYCODE' # або 'CONTRAGENTID'
if id_col in features_to_use:
    features_to_use.remove(id_col)
    
# Також перевір списки, з яких ти клеїш features_to_use, 
# щоб ID не потрапив у log-трансформацію (наприклад, у real_raw_cols)

# 2. Формуємо X, але ID ставимо в індекс
X = df[features_to_use].copy()
X.index = df[id_col] # <--- ОСЬ ГОЛОВНИЙ ТРЮК

y = df["CURR_ACC"]

print(f"FINAL Total Features: {X.shape[1]}")
# Переконайся, що X.columns не містить 'IDENTIFYCODE'
assert id_col not in X.columns




y_pred_val = reg.predict(X_val)

validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_val.index,  # <--- Дістаємо збережені ID
    'True_Value': np.expm1(y_val_log),
    'Predicted': np.expm1(y_pred_val)
})