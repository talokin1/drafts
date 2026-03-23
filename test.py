import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. SPLIT
# =========================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# categorical features (якщо є)
cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()

# =========================================================
# 2. MODEL
# =========================================================

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

# =========================================================
# 3. TRAIN
# =========================================================

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric="mae",
    categorical_feature=cat_cols,
    callbacks=[
        # early stopping
        lambda env: None
    ]
)

# LightGBM early stopping через callbacks (новий API)
from lightgbm import early_stopping, log_evaluation

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="mae",
    categorical_feature=cat_cols,
    callbacks=[
        early_stopping(100),
        log_evaluation(100)
    ]
)

# =========================================================
# 4. PREDICT
# =========================================================

y_pred = model.predict(X_valid)

# якщо у тебе y в log-просторі — розкоментуй:
# y_pred = np.expm1(y_pred)
# y_valid_real = np.expm1(y_valid)

# інакше:
y_valid_real = y_valid

# =========================================================
# 5. METRICS
# =========================================================

mae = mean_absolute_error(y_valid_real, y_pred)
rmse = mean_squared_error(y_valid_real, y_pred, squared=False)
r2 = r2_score(y_valid_real, y_pred)

print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2:   {r2:.4f}")

# =========================================================
# 6. FEATURE IMPORTANCE
# =========================================================

feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop features:")
print(feat_imp.head(20))

# =========================================================
# 7. VALIDATION TABLE
# =========================================================

val_df = pd.DataFrame({
    "y_true": y_valid_real,
    "y_pred": y_pred,
})

val_df["error"] = val_df["y_pred"] - val_df["y_true"]
val_df["abs_error"] = val_df["error"].abs()

print("\nValidation sample:")
print(val_df.head())