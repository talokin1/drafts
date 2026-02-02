import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. ЕТАП: "Selector Model" (Жорстка фільтрація)
# ==========================================
print("--- STEP 1: Training Selector Model (High Regularization) ---")

# Використовуємо сильну L1 регуляризацію (reg_alpha=10), 
# щоб "вбити" шум і знайти реальні драйвери
selector_reg = lgb.LGBMRegressor(
    objective="huber",
    metric="mae",
    alpha=1.5,
    n_estimators=1000,      # Менше дерев для швидкості
    learning_rate=0.05,
    num_leaves=31,
    reg_alpha=10,           # Жорстка чистка
    reg_lambda=10,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Тренуємо на ВСІХ фічах
selector_reg.fit(
    X_train, y_train_log,
    categorical_feature=[c for c in cat_cols if c in X_train.columns],
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

# ==========================================
# 2. ЕТАП: Відбір Топ-70 фічей
# ==========================================
print("\n--- STEP 2: Selecting Top Features ---")

# Витягуємо важливість
feature_imp = pd.DataFrame({
    'Value': selector_reg.feature_importances_,
    'Feature': X_train.columns
}).sort_values(by="Value", ascending=False)

# Беремо Топ-70 (або менше, якщо стільки немає)
TOP_N = 70
selected_features = feature_imp.head(TOP_N)['Feature'].tolist()

print(f"Selected {len(selected_features)} features.")
print(f"Top 5: {selected_features[:5]}")

# Створюємо нові "чисті" датасети (використовуємо .copy() щоб уникнути помилок пам'яті)
X_train_sel = X_train[selected_features].copy()
X_val_sel   = X_val[selected_features].copy()

# Оновлюємо список категоріальних фічей (тільки ті, що вижили)
cat_cols_sel = [c for c in cat_cols if c in selected_features]

# ПЕРЕВІРКА НА ПОМИЛКУ (Safety Check)
assert X_train_sel.shape[0] == len(y_train_log), "X_train and y_train size mismatch!"
assert X_val_sel.shape[0] == len(y_val_log), "X_val and y_val size mismatch!"

# ==========================================
# 3. ЕТАП: "Final Model" (Точне налаштування)
# ==========================================
print("\n--- STEP 3: Training Final Model (Low Regularization) ---")

final_reg = lgb.LGBMRegressor(
    objective="huber",
    metric="mae",
    alpha=1.5,
    n_estimators=5000,      # Більше дерев для точності
    learning_rate=0.03,     # Повільніше навчання для кращої збіжності
    num_leaves=31,
    
    # Зменшуємо регуляризацію, бо сміття ми вже викинули вручну
    reg_alpha=1,            
    reg_lambda=1,
    
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

final_reg.fit(
    X_train_sel, 
    y_train_log,
    categorical_feature=cat_cols_sel,
    eval_set=[(X_val_sel, y_val_log)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
)

# ==========================================
# 4. ЕТАП: Результати
# ==========================================
print("\n--- STEP 4: Evaluation ---")
y_pred_log = final_reg.predict(X_val_sel)

r2 = r2_score(y_val_log, y_pred_log)
mae = mean_absolute_error(y_val_log, y_pred_log)

print(f"FINAL R2 (log space): {r2:.4f}")
print(f"FINAL MAE (log space): {mae:.4f}")

# Графік важливості (для фінальної моделі)
plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=pd.DataFrame({
    'Value': final_reg.feature_importances_,
    'Feature': selected_features
}).sort_values(by="Value", ascending=False).head(20))
plt.title('Final Model - Top 20 Drivers')
plt.tight_layout()
plt.show()