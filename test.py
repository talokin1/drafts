import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    roc_auc_score
)

RANDOM_STATE = 42
TARGET = TARGET_NAME   # твой таргет

# =========================================================
# 1. Подготовка
# =========================================================

df_model = df.copy()

# Бинарный таргет
df_model["has_income"] = (df_model[TARGET] > 0).astype(int)

# Лог-таргет (только для положительных)
df_model["y_log"] = np.log1p(df_model[TARGET])

# Фичи
X = df_model.drop(columns=[TARGET, "has_income", "y_log"])
y_class = df_model["has_income"]
y_reg = df_model["y_log"]

# Стратификация по факту наличия дохода
X_train, X_val, y_class_train, y_class_val = train_test_split(
    X, y_class,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_class
)

# Для регрессии берем только положительные
train_pos_idx = y_class_train == 1
val_pos_idx   = y_class_val == 1

X_train_reg = X_train.loc[train_pos_idx]
X_val_reg   = X_val.loc[val_pos_idx]

y_train_reg = y_reg.loc[X_train_reg.index]
y_val_reg   = y_reg.loc[X_val_reg.index]

# Категориальные
cat_cols = [c for c in X.columns if X[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c]   = X_val[c].astype("category")
    X_train_reg[c] = X_train_reg[c].astype("category")
    X_val_reg[c]   = X_val_reg[c].astype("category")

# =========================================================
# 2. КЛАССИФИКАЦИЯ (есть доход / нет)
# =========================================================

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=32,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train, y_class_train,
    eval_set=[(X_val, y_class_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

p_val = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_class_val, p_val)

# =========================================================
# 3. РЕГРЕССИЯ (только положительные)
# =========================================================

reg = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=32,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train_reg, y_train_reg,
    eval_set=[(X_val_reg, y_val_reg)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

y_pred_log_val_pos = reg.predict(X_val)

# =========================================================
# 4. СБОРКА ФИНАЛЬНОГО ПРЕДСКАЗАНИЯ
# =========================================================

# Предсказание лог-дохода для всех
y_pred_log_all = reg.predict(X_val)

# Финальная формула zero-inflated
y_pred_final = p_val * np.expm1(y_pred_log_all)

# Истинные значения
y_true_final = df_model.loc[X_val.index, TARGET].values

# =========================================================
# 5. МЕТРИКИ
# =========================================================

mae = mean_absolute_error(y_true_final, y_pred_final)
medae = median_absolute_error(y_true_final, y_pred_final)
r2 = r2_score(y_true_final, y_pred_final)

print("=" * 60)
print("CLASSIFICATION AUC:", round(auc, 4))
print("-" * 60)
print("FINAL METRICS (original scale)")
print("MAE   :", round(mae, 2))
print("MedAE :", round(medae, 2))
print("R2    :", round(r2, 4))
print("=" * 60)

# =========================================================
# 6. ВИЗУАЛИЗАЦИЯ
# =========================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_true_final + 1, y_pred_final + 1, alpha=0.3, s=10)
mn = min(y_true_final + 1)
mx = max(y_true_final + 1)
plt.plot([mn, mx], [mn, mx], "r--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True (+1, log axis)")
plt.ylabel("Predicted (+1, log axis)")
plt.title("Zero-Inflated Model: True vs Pred")
plt.show()