import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

TARGET_COL = "CURR_ACC"
RANDOM_STATE = 42

# -----------------------------
# 1) Prepare X, y in LOG SPACE
# -----------------------------
# IMPORTANT: TARGET_COL is already log1p-transformed in your pipeline (as you showed).
# We'll treat it as y_log.
y_log = df[TARGET_COL].astype(float).copy()
X = df.drop(columns=[TARGET_COL]).copy()

# cast categoricals for LightGBM
cat_features = [c for c in X.columns if X[c].dtype.name in ("object", "category")]
for c in cat_features:
    X[c] = X[c].astype("category")

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# --------------------------------------
# 2) Train LGBMRegressor on LOG target
# --------------------------------------
reg = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train,
    y_train_log,
    categorical_feature=cat_features,
    eval_set=[(X_val, y_val_log)],
    eval_metric="l1",  # MAE in log-space (L1)
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
)

# -----------------------------
# 3) Predict (LOG + ORIGINAL)
# -----------------------------
y_pred_log = reg.predict(X_val)

# back to original scale only for reporting/plots
y_val = np.expm1(y_val_log)
y_pred = np.expm1(y_pred_log)

# ---------------------------------
# 4) Metrics: LOG-space (primary)
# ---------------------------------
mae_log = mean_absolute_error(y_val_log, y_pred_log)
r2_log = r2_score(y_val_log, y_pred_log)
medae_log = median_absolute_error(y_val_log, y_pred_log)

# --------------------------------------------
# 5) Metrics: Original scale (robust + normal)
# --------------------------------------------
# Normal MAE will be dominated by tail; we still print it, but do NOT use as main KPI.
mae = mean_absolute_error(y_val, y_pred)
medae = median_absolute_error(y_val, y_pred)

# Relative errors (more stable for heavy-tailed targets)
eps = 1e-9
mape = np.mean(np.abs(y_val - y_pred) / np.maximum(np.abs(y_val), eps))
smape = np.mean(2.0 * np.abs(y_val - y_pred) / np.maximum(np.abs(y_val) + np.abs(y_pred), eps))

print("=" * 60)
print("PRIMARY (LOG-SPACE) METRICS")
print(f"MAE_log    : {mae_log:.5f}")
print(f"MedAE_log  : {medae_log:.5f}")
print(f"R2_log     : {r2_log:.5f}")
print("-" * 60)
print("ORIGINAL-SCALE METRICS (for reference; heavy-tail sensitive)")
print(f"MAE        : {mae:,.2f}")
print(f"MedAE      : {medae:,.2f}")
print(f"MAPE       : {mape:.4f}")
print(f"sMAPE      : {smape:.4f}")
print("=" * 60)

# -----------------------------
# 6) Plots (correct)
# -----------------------------

# 6.1 True vs Pred in LOG SPACE (correctly labeled)
plt.figure(figsize=(8, 6))
plt.scatter(y_val_log, y_pred_log, alpha=0.3, s=10)
mn, mx = float(np.min(y_val_log)), float(np.max(y_val_log))
plt.plot([mn, mx], [mn, mx], "r--")
plt.xlabel("True CURR_ACC (log1p)")
plt.ylabel("Predicted CURR_ACC (log1p)")
plt.title("LGBM Regression: True vs Predicted (log1p space)")
plt.show()

# 6.2 True vs Pred in ORIGINAL SCALE but with LOG-LOG axes (so you see structure)
# NOTE: if y can be 0 after expm1, add +1 for log scale stability
plt.figure(figsize=(8, 6))
plt.scatter(y_val + 1.0, y_pred + 1.0, alpha=0.3, s=10)
mn, mx = float(np.min(y_val + 1.0)), float(np.max(y_val + 1.0))
plt.plot([mn, mx], [mn, mx], "r--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True CURR_ACC (+1), log axis")
plt.ylabel("Predicted CURR_ACC (+1), log axis")
plt.title("LGBM Regression: True vs Predicted (original scale, log-log axes)")
plt.show()

# 6.3 Residuals in LOG SPACE
resid_log = y_val_log - y_pred_log
plt.figure(figsize=(8, 4))
plt.hist(resid_log, bins=80)
plt.xlabel("Residual = True - Pred (log1p)")
plt.ylabel("Count")
plt.title("Residual Distribution (log1p space)")
plt.show()
