import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

reg_model = lgb.LGBMRegressor(
    objective='regression_l1',  # Саме L1 мінімізує MAE
    metric='mae',
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

reg_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

y_pred_log = reg_model.predict(X_val)

mae_log = mean_absolute_error(y_val, y_pred_log)
print(f"MAE on Log scale: {mae_log:.4f}")

y_val_real = np.expm1(y_val)
y_pred_real = np.expm1(y_pred_log)

final_mae = mean_absolute_error(y_val_real, y_pred_real)
print(f"Final MAE (Original Scale): {final_mae:.2f}")