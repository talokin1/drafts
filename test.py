import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    classification_report
)

RANDOM_STATE = 42

# ==========================================================
# 1. TARGET PREPARATION
# ==========================================================

threshold = 0.0
y_binary = (y > threshold).astype(int)

# стратифікація по бінарному таргету
X_train, X_val, y_train, y_val, y_bin_train, y_bin_val = train_test_split(
    X, y, y_binary,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_binary
)

# categorical columns
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

# ==========================================================
# 2. STAGE 1 — CLASSIFICATION (Zero vs Positive)
# ==========================================================

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=128,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train,
    y_bin_train,
    eval_set=[(X_val, y_bin_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(200)]
)

# Predict probability of positive
p_positive_val = clf.predict_proba(X_val)[:, 1]

print("="*60)
print("STAGE 1 — CLASSIFICATION")
print("AUC:", roc_auc_score(y_bin_val, p_positive_val))
print(classification_report(y_bin_val, p_positive_val > 0.5))
print("="*60)

# ==========================================================
# 3. STAGE 2 — REGRESSION (Only Positive Targets)
# ==========================================================

mask_train_pos = y_train > threshold
mask_val_pos = y_val > threshold

X_train_pos = X_train[mask_train_pos]
y_train_pos = y_train[mask_train_pos]

X_val_pos = X_val[mask_val_pos]
y_val_pos = y_val[mask_val_pos]

reg = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.3,
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=256,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train_pos,
    y_train_pos,
    eval_set=[(X_val_pos, y_val_pos)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(300)]
)

# Regression prediction (only for positive validation rows)
y_reg_val_pos = reg.predict(X_val)

# ==========================================================
# 4. FINAL COMBINED PREDICTION
# ==========================================================

# expected value approach
y_pred_val = p_positive_val * y_reg_val_pos

# ==========================================================
# 5. METRICS
# ==========================================================

mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)

# bucket analysis
val_df = pd.DataFrame({
    "true": y_val,
    "pred": y_pred_val
})

val_df["bucket"] = pd.qcut(val_df["true"], q=10, duplicates="drop")

bucket_stats = val_df.groupby("bucket").agg(
    count=("true", "count"),
    mae=("true", lambda x: np.mean(np.abs(x - val_df.loc[x.index, "pred"])))
)

print("="*60)
print("FINAL TWO-STAGE MODEL METRICS")
print("MAE:", round(mae, 2))
print("R2:", round(r2, 4))
print("="*60)
print(bucket_stats)

# ==========================================================
# 6. OPTIONAL — HIGH VALUE ANALYSIS
# ==========================================================

high_threshold = np.percentile(y_val, 90)

mask_high = y_val >= high_threshold

mae_high = mean_absolute_error(
    y_val[mask_high],
    y_pred_val[mask_high]
)

print("="*60)
print("HIGH 10% SEGMENT MAE:", round(mae_high, 2))
print("="*60)
