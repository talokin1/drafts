import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

lgb_reg = LGBMRegressor(
    boosting_type="gbdt",
    objective="regression",
    learning_rate=0.02,
    n_estimators=3000,
    num_leaves=41,
    max_depth=30,
    min_child_samples=20,
    min_child_weight=0.01,
    min_split_gain=0.0,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=100,
    n_jobs=-1,
    importance_type="gain"
)

lgb_reg.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_valid_proc, y_valid)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)


features = X_train_proc.columns
importances = lgb_reg.feature_importances_

feature_importance = (
    pd.DataFrame({
        "features": features,
        "importance": importances
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

feature_importance.head(20)


train_pred = lgb_reg.predict(X_train_proc)

X_train_total = pd.concat(
    [
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        pd.Series(train_pred, name="MODEL_PRED")
    ],
    axis=1
)

mae_train = mean_absolute_error(y_train, train_pred)
rmse_train = mean_squared_error(y_train, train_pred, squared=False)
r2_train = r2_score(y_train, train_pred)

print("TRAIN METRICS")
print("MAE:", round(mae_train, 2))
print("RMSE:", round(rmse_train, 2))
print("R2:", round(r2_train, 4))




valid_pred = lgb_reg.predict(X_valid_proc)

X_valid_total = pd.concat(
    [
        X_valid.reset_index(drop=True),
        y_valid.reset_index(drop=True),
        pd.Series(valid_pred, name="MODEL_PRED")
    ],
    axis=1
)

mae_valid = mean_absolute_error(y_valid, valid_pred)
rmse_valid = mean_squared_error(y_valid, valid_pred, squared=False)
r2_valid = r2_score(y_valid, valid_pred)

print("VALID METRICS")
print("MAE:", round(mae_valid, 2))
print("RMSE:", round(rmse_valid, 2))
print("R2:", round(r2_valid, 4))
