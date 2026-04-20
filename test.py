import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

# ==========================================
# 0. СИМУЛЯЦІЯ ДАНИХ (для відтворюваності)
# ==========================================
np.random.seed(42)
n_samples = 30000
X = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f'feature_{i}' for i in range(10)])

# Генеруємо таргет: 25% нулів, решта - важкий хвіст (Lognormal)
y = np.zeros(n_samples)
is_positive = np.random.rand(n_samples) > 0.25
# Робимо таргет залежним від фічей, щоб моделі було що вчити
y[is_positive] = np.exp(6 + X.loc[is_positive, 'feature_0'] * 1.5 + np.random.randn(is_positive.sum()) * 1.2)
y_series = pd.Series(y)

# ==========================================
# 1. ПРЕПРОЦЕСИНГ ТАРГЕТУ
# ==========================================
THRESHOLD = 50 # Відсікаємо мікрозалишки
y_clean = np.clip(y_series, a_min=0, a_max=None)
y_clean = np.where(y_clean < THRESHOLD, 0, y_clean)

# ==========================================
# 2. СТРАТИФІКАЦІЯ ТА СПЛІТ
# ==========================================
# Розбиваємо ненульові значення на квантилі для надійної стратифікації
non_zero_y = y_clean[y_clean > 0]
percentiles = list(np.percentile(non_zero_y, [25, 50, 75, 90, 95]))
bins = [-np.inf, 0] + percentiles + [np.inf]
y_bins = pd.cut(y_clean, bins=bins, labels=False)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_clean, test_size=0.2, random_state=42, stratify=y_bins
)

# ==========================================
# 3. НАВЧАННЯ МОДЕЛІ TWEEDIE
# ==========================================
reg = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5, # Чим ближче до 2, тим сильніше модель поважає хвіст
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Навчаємо виключно на оригінальних (не логарифмованих) даних!
reg.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="tweedie",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# ==========================================
# 4. ПРЕДИКТ ТА РОЗРАХУНОК МЕТРИК
# ==========================================
y_pred = reg.predict(X_val)
y_pred = np.clip(y_pred, 0, None) # Захист від мікро-від'ємних значень

# Для метрик та візуалізації переходимо в log-простір
y_val_log = np.log1p(y_val)
y_pred_log = np.log1p(y_pred)

print("=" * 60)
print("ОЦІНКА ЯКОСТІ (LOG-SPACE)")
print("=" * 60)
print(f"MAE (log)   : {mean_absolute_error(y_val_log, y_pred_log):.4f}")
print(f"MedAE (log) : {median_absolute_error(y_val_log, y_pred_log):.4f}")
print(f"R² Score    : {r2_score(y_val_log, y_pred_log):.4f}")
print("-" * 60)

# Додаткова бізнес-метрика: Точність класифікації "Нуль vs Не нуль"
val_is_zero = (y_val == 0)
pred_is_zero = (y_pred < THRESHOLD)
accuracy_zeros = np.mean(val_is_zero == pred_is_zero)
print(f"Точність розпізнавання нулів/пустих рахунків: {accuracy_zeros:.1%}")
print("=" * 60)

# ==========================================
# 5. ВІЗУАЛІЗАЦІЯ РОЗПОДІЛІВ
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Графік 1: True vs Predicted (Scatter)
axes[0].scatter(y_val_log, y_pred_log, alpha=0.2, s=15, color='royalblue', label='Predictions')
mn, mx = 0, max(y_val_log.max(), y_pred_log.max()) + 1
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Ideal Fit')
axes[0].set_title("Scatter: Predicted vs Actual (log1p space)", fontsize=14)
axes[0].set_xlabel("True Target $log(1 + Y)$")
axes[0].set_ylabel("Predicted Target $log(1 + \hat{Y})$")
axes[0].legend()

# Графік 2: Kernel Density Estimation (Збіг розподілів)
sns.kdeplot(y_val_log, color='crimson', label='True Distribution', fill=True, alpha=0.1, ax=axes[1], linewidth=2)
sns.kdeplot(y_pred_log, color='navy', label='Tweedie Prediction', linestyle='--', ax=axes[1], linewidth=2)
axes[1].set_title("Distribution Match: True vs Predicted", fontsize=14)
axes[1].set_xlabel("$log(1 + Y)$")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.show()