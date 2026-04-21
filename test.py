# Отримуємо ймовірності замість жорстких 0/1
val_class_proba = clf.predict_proba(X_val)[:, 1]

# Шукаємо найкращий поріг (наївний підхід для прикладу)
best_threshold = 0.5
best_mae = float('inf')

for thresh in np.arange(0.1, 0.9, 0.05):
    # Тестуємо різні пороги
    temp_class_preds = (val_class_proba >= thresh).astype(int)
    temp_y_pred = np.where(temp_class_preds == 1, val_reg_preds, 0)
    temp_y_pred_log = np.log1p(temp_y_pred)
    
    current_mae = mean_absolute_error(y_val_final_log, temp_y_pred_log)
    if current_mae < best_mae:
        best_mae = current_mae
        best_threshold = thresh

print(f"Оптимальний поріг класифікації: {best_threshold:.2f}")

# Використовуємо знайдений поріг
val_class_preds = (val_class_proba >= best_threshold).astype(int)
y_pred_final = np.where(val_class_preds == 1, val_reg_preds, 0)

python main.py --input input.csv --output company_results.csv --proxy-file proxies.txt
