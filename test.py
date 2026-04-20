import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, r2_score, median_absolute_error, 
                             roc_auc_score, classification_report)

RANDOM_STATE = 42
THRESHOLD = 50  # Поріг відсічення "пустих" клієнтів

# ==========================================
# 1. ГЛОБАЛЬНИЙ СПЛІТ ДАНИХ
# ==========================================
# Прибираємо від'ємні значення
y_clean = np.clip(y, a_min=0, a_max=None)

# Створюємо бінарний таргет для класифікації та стратифікації
y_binary = (y_clean >= THRESHOLD).astype(int)

# Робимо спліт, стратифікуючи за наявністю грошей
X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    X, y_clean, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
)

# Обробка категоріальних фічей
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")


# ==========================================
# 2. STAGE 1: КЛАСИФІКАТОР (Гроші є чи ні?)
# ==========================================
print("Навчання Stage 1: Classifier...")
y_train_clf = (y_train_raw >= THRESHOLD).astype(int)
y_val_clf = (y_val_raw >= THRESHOLD).astype(int)

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    categorical_feature=cat_cols,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train, y_train_clf,
    eval_set=[(X_val, y_val_clf)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)


# ==========================================
# 3. STAGE 2: РЕГРЕСОР (Скільки грошей?)
# ==========================================
print("Навчання Stage 2: Regressor...")
# Фільтруємо ТІЛЬКИ тих клієнтів, у кого є гроші (за трейном)
mask_train_active = y_train_raw >= THRESHOLD
X_train_reg = X_train[mask_train_active]
y_train_reg_log = np.log1p(y_train_raw[mask_train_active]) # Повертаємося до log1p!

# Для валідації регресора теж беремо тільки активних
mask_val_active = y_val_raw >= THRESHOLD
X_val_reg = X_val[mask_val_active]
y_val_reg_log = np.log1p(y_val_raw[mask_val_active])

reg = lgb.LGBMRegressor(
    objective="huber", # Huber або "regression" (MSE). Huber краще ігнорує супер-викиди
    alpha=1.5,         # Параметр для Huber
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    categorical_feature=cat_cols,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train_reg, y_train_reg_log,
    eval_set=[(X_val_reg, y_val_reg_log)],
    eval_metric="l1", # Моніторимо MAE
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
)


# ==========================================
# 4. КОМБІНОВАНИЙ ПРЕДИКТ (HARD ROUTING)
# ==========================================
# 1. Передбачаємо клас (0 або 1) для ВСІЄЇ валідаційної вибірки
val_class_preds = clf.predict(X_val)

# 2. Передбачаємо суму для ВСІЄЇ валідаційної вибірки (в логарифмі)
val_reg_preds_log = reg.predict(X_val)
val_reg_preds = np.expm1(val_reg_preds_log)

# 3. Маршрутизація: якщо класифікатор сказав 0, ставимо 0. Інакше - прогноз регресора.
y_pred_final = np.where(val_class_preds == 1, val_reg_preds, 0)


# ==========================================
# 5. ОЦІНКА ЯКОСТІ
# ==========================================
y_val_final_log = np.log1p(y_val_raw)
y_pred_final_log = np.log1p(y_pred_final)

mae_log = mean_absolute_error(y_val_final_log, y_pred_final_log)
r2_log = r2_score(y_val_final_log, y_pred_final_log)
medae_log = median_absolute_error(y_val_final_log, y_pred_final_log)

eps = 1e-9
mape = np.mean(np.abs(y_val_raw - y_pred_final) / np.maximum(np.abs(y_val_raw), eps))

print("=" * 60)
print("CLASSIFIER METRICS (Stage 1)")
print(f"ROC-AUC: {roc_auc_score(y_val_clf, clf.predict_proba(X_val)[:, 1]):.4f}")
print("-" * 60)
print("COMBINED PIPELINE METRICS (LOG-SPACE)")
print(f"MAE_log   : {mae_log:.5f}")
print(f"MedAE_log : {medae_log:.5f}")
print(f"R2_log    : {r2_log:.5f}")
print("=" * 60)

# Візуалізація
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_val_final_log, y_pred_final_log, alpha=0.2, s=15, color='royalblue')
mn, mx = 0, float(np.max(y_val_final_log)) + 1
axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=2)
axes[0].set_xlabel("True CURR_ACC (log1p)")
axes[0].set_ylabel("Predicted CURR_ACC (log1p)")
axes[0].set_title("Two-Stage Model: Scatter Plot")

sns.kdeplot(y_val_final_log, color='crimson', label='True Distribution', fill=True, alpha=0.1, ax=axes[1], linewidth=2)
sns.kdeplot(y_pred_final_log, color='navy', label='Two-Stage Prediction', linestyle='--', ax=axes[1], linewidth=2)
axes[1].set_title("Two-Stage Model: Distribution Match")
axes[1].set_xlabel("log(1 + CURR_ACC)")
axes[1].legend()

plt.tight_layout()
plt.show()