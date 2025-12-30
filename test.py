LGBMRegressor(
    boosting_type="gbdt",
    objective="regression",
    learning_rate=0.03,
    n_estimators=2000,

    max_depth=8,
    num_leaves=31,

    min_child_samples=50,
    min_child_weight=0.1,
    min_split_gain=0.01,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=100,
    n_jobs=-1,
    importance_type="gain"
)
