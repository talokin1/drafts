# --- ВАШ ОРИГІНАЛЬНИЙ БЛОК РЕГРЕСІЇ (БЕЗ ЗМІН) ---

df_ = df.copy()
df_ = df_[(df_["CURR_ACC"] > 0.05) & (df_["CURR_ACC"] < df_["CURR_ACC"].quantile(0.97))]
df_["CURR_ACC"] = np.log1p(df_["CURR_ACC"])

# ... далі йде ваш train_test_split на df_ ...
# ... далі йде reg.fit(...) ...




# --- ІНФЕРЕНС: ДВОСТУПІНЧАТИЙ ПРОГНОЗ ---

# Для чесної валідації беремо вибірку з нулями (з Кроку 1)
X_test_scoring = X_val_clf.copy()
# Справжній прибуток (без логарифмів і усічень), витягуємо з оригінального df
y_test_true = df.loc[X_test_scoring.index, "CURR_ACC"] 

# Етап 1: Класифікатор приймає рішення (0 або 1)
# Можна використовувати predict(), або predict_proba() > threshold для тонкого налаштування
is_profitable = clf.predict(X_test_scoring)

# Етап 2: Регресор прогнозує суму для ВСІХ
log_profit = reg.predict(X_test_scoring)
actual_profit = np.expm1(log_profit)

# Етап 3: Математичне об'єднання E[Y] = P(Y>0) * E[Y|Y>0]
final_predicted_profit = is_profitable * actual_profit

# --- ФОРМУВАННЯ ЗВІТУ ---
validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_test_scoring.index,
    'True_Value': y_test_true, 
    'Predicted': final_predicted_profit
})

# Рахуємо метрики на фінальних значеннях
mae = mean_absolute_error(validation_results['True_Value'], validation_results['Predicted'])
print(f"Combined MAE: {mae}")

# ... далі запускаєте ваш код створення Excel (validation_results.copy()...)