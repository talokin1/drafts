y_train_bin = (y_train > 0).astype(int)
y_valid_bin = (y_valid > 0).astype(int)


import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

lgb_stage1 = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    class_weight='balanced',
    learning_rate=0.02,
    max_depth=30,
    min_child_samples=20,
    min_child_weight=0.01,
    min_split_gain=0.0,
    num_leaves=41,
    n_estimators=3000,
    subsample=1.0,
    colsample_bytree=1.0,
    subsample_for_bin=200000,
    subsample_freq=0,
    random_state=100,
    n_jobs=-1,
    importance_type='gain',
    verbose=1
)


lgb_stage1.fit(
    X_train_proc,
    y_train_bin,
    eval_set=[(X_valid_proc, y_valid_bin)],
    eval_metric='f1',
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

features_stage1 = X_train_proc.columns
importances_stage1 = lgb_stage1.feature_importances_

feature_importance_stage1 = (
    pd.DataFrame({
        'importance': importances_stage1,
        'features': features_stage1
    })
    .sort_values('importance', ascending=False)
    .reset_index(drop=True)
)





forecast_train_stage1 = pd.DataFrame(
    lgb_stage1.predict_proba(X_train_proc),
    columns=['ZERO_INCOME', 'NON_ZERO']
)

forecast_train_stage1 = forecast_train_stage1[['NON_ZERO']]

X_train_stage1_total = pd.concat(
    [
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        forecast_train_stage1.reset_index(drop=True)
    ],
    axis=1
)




forecast_valid_stage1 = pd.DataFrame(
    lgb_stage1.predict_proba(X_valid_proc),
    columns=['ZERO_INCOME', 'NON_ZERO']
)

forecast_valid_stage1 = forecast_valid_stage1[['NON_ZERO']]

X_valid_stage1_total = pd.concat(
    [
        X_valid.reset_index(drop=True),
        y_valid.reset_index(drop=True),
        forecast_valid_stage1.reset_index(drop=True)
    ],
    axis=1
)

