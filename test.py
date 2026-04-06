# --- НОВИЙ БЛОК: ТРЕНУВАННЯ КЛАСИФІКАТОРА ---

# 1. Створюємо бінарний таргет на повному датасеті
y_clf = (df["CURR_ACC"] > 0.05).astype(int)
X_clf = df[features_to_use].copy() # features_to_use з вашого коду

# 2. Спліт для класифікатора (ВАЖЛИВО: зберігаємо цей X_val_clf для фінального звіту!)
X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

# Форматування категорій для LightGBM
cat_cols = [c for c in X_train_clf.columns if X_train_clf[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train_clf[c] = X_train_clf[c].astype("category")
    X_val_clf[c] = X_val_clf[c].astype("category")

# 3. Навчання класифікатора
clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    class_weight='balanced', # Компенсуємо дисбаланс нулів
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train_clf, y_train_clf,
    eval_set=[(X_val_clf, y_val_clf)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)
print("Класифікатор натреновано!")





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