import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

RANDOM_STATE = 42
THRESHOLD = 50 # Бізнес-поріг. Суми менші за 50 грн вважаємо "пустими" рахунками

# ==========================================
# 1. ПРЕПРОЦЕСИНГ ТАРГЕТУ (Критично для Tweedie)
# ==========================================
# Відсікаємо від'ємні значення (овердрафти не є пасивами)
y_clean = np.clip(y, a_min=0, a_max=None)

# Обнуляємо мікро-залишки, формуючи чіткий zero-inflated розподіл
y_clean = np.where(y_clean < THRESHOLD, 0, y_clean)


# ==========================================
# 2. БЕЗПЕЧНА СТРАТИФІКАЦІЯ
# ==========================================
# Робимо кастомні біни, щоб гарантовано відокремити нулі від 'багатих'
non_zero_y = y_clean[y_clean > 0]
percentiles = list(np.percentile(non_zero_y, [25, 50, 75, 90]))
bins = [-np.inf, 0] + percentiles + [np.inf]

y_bins = pd.cut(y_clean, bins=bins, labels=False)


# ==========================================
# 3. SPLIT ТА ПІДГОТОВКА ФІЧЕЙ
# ==========================================
# ЗВЕРНИ УВАГУ: Передаємо y_clean (оригінальний масштаб), а не логарифм!
X_train, X_val, y_train, y_val = train_test_split(
    X, y_clean, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bins
)

X_train_final = X_train.copy()
X_val_final = X_val.copy()

cat_cols = [c for c in X_train_final.columns if X_train_final[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train_final[c] = X_train_final[c].astype("category")
    X_val_final[c] = X_val_final[c].astype("category")


# ==========================================
# 4. TWEEDIE АРХІТЕКТУРА
# ==========================================
reg = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5, # Гіперпараметр p. 1.5 - класика для балансів.
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    categorical_feature=cat_cols, # Вказуємо явно для уникнення ворнінгів
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Навчаємо на ОРИГІНАЛЬНИХ значеннях
reg.fit(
    X_train_final,
    y_train,
    eval_set=[(X_val_final, y_val)],
    eval_metric="tweedie", # Функція втрат враховує дисперсію хвостів
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
)


# ==========================================
# 5. ОЦІНКА ТА АНАЛІЗ
# ==========================================
# Предикт повертає значення в оригінальному масштабі (вбудований exp під капотом)
y_pred = reg.predict(X_val_final)

# Оскільки Твіді може інколи видавати мікро-від'ємні предикати через наближення:
y_pred = np.clip(y_pred, 0, None)

# Для того, щоб адекватно ПОРІВНЯТИ метрики з твоєю минулою моделлю, 
# ми тимчасово логарифмуємо результати суто для розрахунку R2 та побудови графіка
y_val_log = np.log1p(y_val)
y_pred_log = np.log1p(y_pred)

mae_log = mean_absolute_error(y_val_log, y_pred_log)
r2_log = r2_score(y_val_log, y_pred_log)
medae_log = median_absolute_error(y_val_log, y_pred_log)

mae_orig = mean_absolute_error(y_val, y_pred)
medae_orig = median_absolute_error(y_val, y_pred)

print("=" * 60)
print("METRICS IN LOG-SPACE (for comparing with previous model)")
print(f"MAE_log   : {mae_log:.5f}")
print(f"MedAE_log : {medae_log:.5f}")
print(f"R2_log    : {r2_log:.5f}")
print("-" * 60)
print("METRICS IN ORIGINAL SPACE")
print(f"MAE_orig  : {mae_orig:,.2f}")
print(f"MedAE_orig: {medae_orig:,.2f}")
print("=" * 60)

# Візуалізація
plt.figure(figsize=(8, 6))
plt.scatter(y_val_log, y_pred_log, alpha=0.3, s=10)
mn, mx = float(np.min(y_val_log)), float(np.max(y_val_log))
plt.plot([mn, mx], [mn, mx], "r--")
plt.xlabel("True CURR_ACC (log1p)")
plt.ylabel("Predicted CURR_ACC (log1p)")
plt.title("Tweedie Regression: True vs Predicted (Visualized in log1p space)")
plt.show()