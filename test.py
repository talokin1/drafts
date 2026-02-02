import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 0. ЕТАП: Підготовка даних (Safety First)
# ==========================================
print("--- STEP 0: Preparing Data & Splitting ---")

# 1. Формуємо X та y із вашого поточного датафрейму df
# (Припускаємо, що df вже містить всі потрібні колонки features_to_use)
# Якщо features_to_use ще не визначено, беремо всі колонки крім таргета
target_col = "CURR_ACC"

# Видаляємо таргет з фічей, якщо він там випадково є
X = df.drop(columns=[target_col], errors='ignore').copy()

# 2. Формуємо таргет (логарифмуємо)
# Важливо: якщо ви вже фільтрували нулі, то log1p ок. 
# Якщо ні - clip(0) страхує від помилок.
y = np.log1p(df[target_col].clip(lower=0))

# 3. Переконуємось, що категорії мають правильний тип
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_features:
    X[c] = X[c].astype('category')

print(f"Total samples: {X.shape[0]}")
print(f"Total features: {X.shape[1]}")

# 4. РОБИМО СПЛІТ (Це виправляє вашу помилку!)
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Перевірка розмірності (щоб точно не впало)
assert len(X_train) == len(y_train_log), "Train sizes mismatch!"
assert len(X_val) == len(y_val_log), "Val sizes mismatch!"
print("Data split successful.")

# ==========================================
# 1. ЕТАП: "Selector Model" (Жорстка фільтрація)
# ==========================================
print("\n--- STEP 1: Training Selector Model (High Regularization) ---")

selector_reg = lgb.LGBMRegressor(
    objective="huber",      # Стійкість до викидів
    metric="mae",
    alpha=1.5,
    n_estimators=1000,      # Швидкий прогін
    learning_rate=0.05,
    num_leaves=31,
    
    # Жорстка регуляризація для відсіву сміття
    reg_alpha=10,           
    reg_lambda=10,
    
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

selector_reg.fit(
    X_train, y_train_log,
    categorical_feature=cat_features,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

# ==========================================
# 2. ЕТАП: Відбір Топ-70 фічей
# ==========================================
print("\n--- STEP 2: Selecting Top Features ---")

feature_imp = pd.DataFrame({
    'Value': selector_reg.feature_importances_,
    'Feature': X_train.columns
}).sort_values(by="Value", ascending=False)

# Беремо Топ-70
TOP_N = 70
selected_features = feature_imp.head(TOP_N)['Feature'].tolist()

print(f"Selected top {len(selected_features)} features.")
print(f"Top 5 drivers: {selected_features[:5]}")

# Оновлюємо датасети (залишаємо тільки важливе)
X_train_sel = X_train[selected_features].copy()
X_val_sel   = X_val[selected_features].copy()

# Оновлюємо список категорій (тільки ті, що залишились)
cat_features_sel = [c for c in cat_features if c in selected_features]

# ==========================================
# 3. ЕТАП: "Final Model" (Точне налаштування)
# ==========================================
print("\n--- STEP 3: Training Final Model (Low Regularization) ---")

final_reg = lgb.LGBMRegressor(
    objective="huber",
    metric="mae",
    alpha=1.5,
    n_estimators=5000,      # Довге навчання для якості
    learning_rate=0.03,     # Повільніше і точніше
    num_leaves=31,
    
    # Зменшуємо регуляризацію, бо сміття вже немає
    reg_alpha=1,            
    reg_lambda=1,
    
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

final_reg.fit(
    X_train_sel, 
    y_train_log,
    categorical_feature=cat_features_sel,
    eval_set=[(X_val_sel, y_val_log)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
)

# ==========================================
# 4. ЕТАП: Оцінка результатів
# ==========================================
print("\n--- STEP 4: Evaluation ---")
y_pred_log = final_reg.predict(X_val_sel)

r2 = r2_score(y_val_log, y_pred_log)
mae = mean_absolute_error(y_val_log, y_pred_log)

print(f"FINAL R2 (log space): {r2:.4f}")
print(f"FINAL MAE (log space): {mae:.4f}")

# Графік важливості
plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=pd.DataFrame({
    'Value': final_reg.feature_importances_,
    'Feature': selected_features
}).sort_values(by="Value", ascending=False).head(20))
plt.title('Final Model - Top 20 Drivers')
plt.tight_layout()
plt.show()