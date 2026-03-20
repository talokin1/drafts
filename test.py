import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score
)

# =========================================================
# 0. CONFIG
# =========================================================
RANDOM_STATE = 42
TARGET_NAME = "TRXS_INCOME"   # <-- заміни на свою назву таргету
ID_COL = "IDENTIFYCODE"       # <-- якщо ID у тебе в індексі, це теж ок

# =========================================================
# 1. COPY DATA
# =========================================================
df_model = df.copy()

# Якщо ID лежить в індексі, можна цей блок не чіпати.
# Якщо ID окремою колонкою, лишаємо його для validation_results,
# але не подаємо в модель.
if ID_COL in df_model.columns:
    ids_all = df_model[ID_COL].copy()
else:
    ids_all = df_model.index.copy()

# =========================================================
# 2. FILTER TARGET > 0
#    (бо ти сказав, що нулі для бізнесу неважливі)
# =========================================================
df_model = df_model[df_model[TARGET_NAME] > 0].copy()

# Перевірка
print(f"Rows after target > 0 filter: {len(df_model):,}")

# =========================================================
# 3. DEFINE X / y
# =========================================================
drop_cols = [TARGET_NAME]

# Якщо ID окремою колонкою і не треба в модель
if ID_COL in df_model.columns:
    drop_cols.append(ID_COL)

X = df_model.drop(columns=drop_cols).copy()
y = df_model[TARGET_NAME].copy()

# Лог-таргет
y_log = np.log1p(y)

# Для stratify по безперервному таргету
# Якщо qcut дає мало унікальних бакетів, можна зменшити q
y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

# =========================================================
# 4. TRAIN / VALID SPLIT
# =========================================================
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_bins
)

# Окремо тримаємо original-scale target для зручності
y_train = np.expm1(y_train_log)
y_val = np.expm1(y_val_log)

# ID для validation_results
ids_train = X_train.index
ids_val = X_val.index

# =========================================================
# 5. CATEGORICAL FEATURES
# =========================================================
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

    # Вирівнюємо категорії validation до train
    X_val[c] = X_val[c].cat.set_categories(X_train[c].cat.categories)

X_train_final = X_train.copy()
X_val_final = X_val.copy()

print(f"Train shape: {X_train_final.shape}")
print(f"Valid shape: {X_val_final.shape}")
print(f"Categorical cols: {len(cat_cols)}")

# =========================================================
# 6. MODEL
# =========================================================
reg = lgb.LGBMRegressor(
    objective="huber",         # можна ще спробувати 'regression', 'mae', 'tweedie'
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=180,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train_final,
    y_train_log,
    eval_set=[(X_val_final, y_val_log)],
    eval_metric="l1",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=True),
        lgb.log_evaluation(200)
    ]
)

# =========================================================
# 7. PREDICTIONS
# =========================================================
# Прогноз у log-space
y_pred_log = reg.predict(X_val_final)

# Прогноз в original space
y_pred = np.expm1(y_pred_log)

# Захист від негативних значень після зворотного перетворення
y_pred = np.clip(y_pred, a_min=0, a_max=None)

# =========================================================
# 8. OPTIONAL CALIBRATION
#    Калібрування прогнозу в log-space
# =========================================================
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

poly_model.fit(y_pred_log.reshape(-1, 1), y_val_log)

y_pred_log_cal = poly_model.predict(y_pred_log.reshape(-1, 1))
y_pred_cal = np.expm1(y_pred_log_cal)
y_pred_cal = np.clip(y_pred_cal, a_min=0, a_max=None)

# =========================================================
# 9. METRICS
# =========================================================
# --- LOG SPACE ---
mae_log = mean_absolute_error(y_val_log, y_pred_log)
medae_log = median_absolute_error(y_val_log, y_pred_log)
r2_log = r2_score(y_val_log, y_pred_log)

# --- ORIGINAL SPACE ---
mae = mean_absolute_error(y_val, y_pred)
mae_cal = mean_absolute_error(y_val, y_pred_cal)
medae = median_absolute_error(y_val, y_pred)
r2_orig = r2_score(y_val, y_pred)

eps = 1e-9
mape = np.mean(np.abs(y_val - y_pred) / np.maximum(np.abs(y_val), eps))
smape = np.mean(2.0 * np.abs(y_val - y_pred) / np.maximum(np.abs(y_val) + np.abs(y_pred), eps))

print("=" * 70)
print("PRIMARY (LOG-SPACE) METRICS")
print(f"MAE_log    : {mae_log:.5f}")
print(f"MedAE_log  : {medae_log:.5f}")
print(f"R2_log     : {r2_log:.5f}")

print("=" * 70)
print("ORIGINAL-SCALE METRICS")
print(f"MAE        : {mae:,.2f}")
print(f"MAE_cal    : {mae_cal:,.2f}")
print(f"MedAE      : {medae:,.2f}")
print(f"R2_orig    : {r2_orig:.5f}")
print(f"MAPE       : {mape:.4f}")
print(f"sMAPE      : {smape:.4f}")
print("=" * 70)

# =========================================================
# 10. VALIDATION RESULTS TABLE
# =========================================================
validation_results = pd.DataFrame({
    "IDENTIFYCODE": ids_val,
    "TRUE_VALUE": y_val,
    "PREDICTED": y_pred,
    "PREDICTED_CAL": y_pred_cal,
    "ABS_ERROR": np.abs(y_val - y_pred),
    "ABS_ERROR_CAL": np.abs(y_val - y_pred_cal),
    "REL_ERROR": np.abs(y_val - y_pred) / np.maximum(y_val, eps),
    "REL_ERROR_CAL": np.abs(y_val - y_pred_cal) / np.maximum(y_val, eps),
    "TRUE_LOG": y_val_log,
    "PRED_LOG": y_pred_log,
    "PRED_LOG_CAL": y_pred_log_cal
}).sort_values("ABS_ERROR", ascending=False)

display(validation_results.head(20))

# =========================================================
# 11. FEATURE IMPORTANCE
# =========================================================
feature_importance = pd.DataFrame({
    "feature": X_train_final.columns,
    "importance_gain": reg.booster_.feature_importance(importance_type="gain"),
    "importance_split": reg.booster_.feature_importance(importance_type="split")
}).sort_values("importance_gain", ascending=False)

display(feature_importance.head(30))

# =========================================================
# 12. ERROR BY TARGET BUCKET
# =========================================================
validation_results["TRUE_BUCKET"] = pd.qcut(
    validation_results["TRUE_VALUE"],
    q=10,
    duplicates="drop"
)

bucket_metrics = validation_results.groupby("TRUE_BUCKET").agg(
    cnt=("TRUE_VALUE", "size"),
    true_mean=("TRUE_VALUE", "mean"),
    pred_mean=("PREDICTED", "mean"),
    mae=("ABS_ERROR", "mean"),
    medae=("ABS_ERROR", "median"),
    mape=("REL_ERROR", "mean")
).reset_index()

display(bucket_metrics)