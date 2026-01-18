import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Фільтруємо тренувальні дані (беремо тільки тих, кого ми вважаємо "цільовими")
# Важливо: використовуємо реальні мітки (y_train_cls), а не передбачення класифікатора,
# щоб регресор вчився на "чистій" істині.
mask_vip_train = y_train_cls == 1 
X_train_reg = X_train[mask_vip_train]
y_train_reg_log = y_train_log[mask_vip_train] # Нагадую, таргет логарифмований!

print(f"Навчаємо регресію на {len(X_train_reg)} об'єктах (VIP сегмент)...")

# 2. Ініціалізуємо та вчимо регресор
regressor = lgb.LGBMRegressor(
    objective='regression',
    metric='mae',        # MAE краще сприймає викиди, ніж MSE
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

regressor.fit(
    X_train_reg, 
    y_train_reg_log, 
    categorical_feature=cat_features
)
print("Регресор навчено")


def final_predict(X_input, classifier, regressor, baseline_value=0):
    """
    X_input: вхідні дані
    classifier: навчена модель класифікації
    regressor: навчена модель регресії
    baseline_value: скільки прогнозуємо для "дрібних" (можна поставити 0 або медіану дрібних)
    """
    # 1. Етап класифікації
    # Прогнозуємо клас (0 або 1)
    pred_classes = classifier.predict(X_input)
    
    # 2. Етап регресії
    # Робимо прогноз регресії для ВСІХ (це швидше векторизовано, ніж фільтрувати)
    pred_log = regressor.predict(X_input)
    pred_amounts = np.expm1(pred_log) # Повертаємо з логарифма в гроші
    
    # 3. Комбінація (Hurdle Logic)
    # Якщо клас 1 -> беремо прогноз регресії
    # Якщо клас 0 -> беремо baseline_value
    final_predictions = np.where(pred_classes == 1, pred_amounts, baseline_value)
    
    return final_predictions

# --- Перевірка на тестових даних ---
final_preds = final_predict(X_test, clf, regressor, baseline_value=500) # Наприклад, дрібним ставимо 500 грн

# Оцінка фінального результату
final_mae = mean_absolute_error(np.expm1(y_test_log), final_preds) # Порівнюємо з реальними грошима
final_r2 = r2_score(np.expm1(y_test_log), final_preds)

print(f"\n--- ФІНАЛЬНИЙ РЕЗУЛЬТАТ (Two-Stage Model) ---")
print(f"MAE: {final_mae:.2f} грн")
print(f"R2: {final_r2:.4f}")