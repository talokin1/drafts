import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

lgb_reg = LGBMRegressor(
    boosting_type="gbdt",
    objective="regression",
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=30,
    num_leaves=41,
    min_child_samples=20,
    min_child_weight=0.01,
    subsample=1.0,
    subsample_freq=0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=100,
    n_jobs=-1,
    importance_type="gain"
)

lgb_reg.fit(
    X_train_prep,
    y_train,
    eval_set=[(X_valid_prep, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)


y_valid_pred_log = lgb_reg.predict(X_valid_prep)
y_valid_pred = np.expm1(y_valid_pred_log)
y_valid_true = np.expm1(y_valid)
rmse = mean_squared_error(y_valid_true, y_valid_pred, squared=False)
mae = mean_absolute_error(y_valid_true, y_valid_pred)

print(f"RMSE: {rmse:,.2f}")
print(f"MAE : {mae:,.2f}")


feature_importance = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": lgb_reg.feature_importances_
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

