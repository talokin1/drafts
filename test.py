import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Підготовка X та y
target_col = 'CURR_ACC'

# Видаляємо таргет та "сміттєві" колонки з X
drop_cols = [target_col, 'TERM_DEPOSITS', 'IDENTIFYCODE', 'FIRM_NAME'] 
# Додайте сюди інші ID, якщо вони залишились

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target_col]

# 2. Логарифмування таргету (Критично для грошей!)
# Використовуємо log1p (log(1+x)), щоб не отримати -inf від нуля
y_log = np.log1p(y)

# 3. Спліт (Припускаємо, що це Snapshot, тому random. Якщо ні - робіть по часу)
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# 4. Створення Dataset для LGB
# Вказуємо, які колонки є категоріальними (КВЕДи, ОПФГ)
cat_features = [c for c in X.columns if X[c].dtype.name == 'category']
print(f"Категоріальні фічі: {cat_features}")

train_data = lgb.Dataset(X_train, label=y_train_log, categorical_feature=cat_features)
test_data = lgb.Dataset(X_test, label=y_test_log, reference=train_data, categorical_feature=cat_features)

# 5. Параметри (Базові, без тюнінгу)
params = {
    'objective': 'regression',
    'metric': 'rmse',        # Оптимізуємо помилку на логарифмах
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1             # Всі ядра процесора
}

# 6. Тренування
print("Починаємо навчання...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# 7. Валідація (Повертаємось з логарифмів у реальні гроші!)
preds_log = model.predict(X_test)
preds_real = np.expm1(preds_log) # Обернена до log1p
y_test_real = np.expm1(y_test_log)

# Метрики
mae = mean_absolute_error(y_test_real, preds_real)
r2 = r2_score(y_test_real, preds_real)

print(f"\n--- Результати ---")
print(f"MAE (Середня помилка в валюті): {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# 8. Feature Importance (Що вплинуло найбільше?)
lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10, 6))