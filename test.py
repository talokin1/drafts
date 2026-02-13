import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score
)

RANDOM_STATE = 42

# =====================================================
# 1️⃣ TRAIN / VAL SPLIT (STRATIFIED BY TARGET)
# =====================================================

y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_bins
)

# лог-простір
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)

# =====================================================
# 2️⃣ CATEGORICAL HANDLING
# =====================================================

cat_cols = [c for c in X_train.columns 
            if X_train[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c]   = X_val[c].astype("category")

# =====================================================
# 3️⃣ LIGHTGBM
# =====================================================

reg = lgb.LGBMRegressor(
    objective="huber",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.8,
    categorical_feature=cat_cols,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(200)]
)

# =====================================================
# 4️⃣ BASE PREDICTIONS
# =====================================================

y_pred_log = reg.predict(X_val)

# =====================================================
# 5️⃣ POST-CALIBRATION (ISOTONIC)
# =====================================================

iso = IsotonicRegression(out_of_bounds="clip")

iso.fit(y_pred_log, y_val_log)

y_pred_log_cal = iso.predict(y_pred_log)

# =====================================================
# 6️⃣ BACK TO ORIGINAL SCALE
# =====================================================

y_val_orig      = np.expm1(y_val_log)
y_pred_orig     = np.expm1(y_pred_log)
y_pred_cal_orig = np.expm1(y_pred_log_cal)

# =====================================================
# 7️⃣ METRICS
# =====================================================

print("="*60)
print("LOG SPACE METRICS")
print("="*60)

print("MAE_log      :", mean_absolute_error(y_val_log, y_pred_log))
print("MAE_log_cal  :", mean_absolute_error(y_val_log, y_pred_log_cal))
print("R2_log       :", r2_score(y_val_log, y_pred_log))

print("="*60)
print("ORIGINAL SCALE METRICS")
print("="*60)

print("MAE base     :", mean_absolute_error(y_val_orig, y_pred_orig))
print("MAE calibrated:", mean_absolute_error(y_val_orig, y_pred_cal_orig))
print("MedAE        :", median_absolute_error(y_val_orig, y_pred_orig))

# =====================================================
# 8️⃣ VALIDATION RESULTS DF
# =====================================================

validation_results = pd.DataFrame({
    "IDENTIFYCODE": X_val.index,
    "True_Value": y_val_orig,
    "Predicted_Base": y_pred_orig,
    "Predicted_Calibrated": y_pred_cal_orig,
    "Abs_Error_Base": np.abs(y_val_orig - y_pred_orig),
    "Abs_Error_Cal": np.abs(y_val_orig - y_pred_cal_orig)
})

# =====================================================
# 9️⃣ DIAGNOSTICS
# =====================================================

plt.figure(figsize=(6,6))
plt.scatter(y_val_log, y_pred_log, alpha=0.3, s=10)
plt.plot([y_val_log.min(), y_val_log.max()],
         [y_val_log.min(), y_val_log.max()],
         "r--")
plt.title("Log Space: Base Model")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_val_log, y_pred_log_cal, alpha=0.3, s=10)
plt.plot([y_val_log.min(), y_val_log.max()],
         [y_val_log.min(), y_val_log.max()],
         "r--")
plt.title("Log Space: Calibrated")
plt.show()

residuals = y_val_log - y_pred_log
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=80)
plt.title("Residual Distribution (log space)")
plt.show()
