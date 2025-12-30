STAGE1_THRESHOLD = 0.5

train_mask = y_train_bin == 1
valid_mask = y_valid_bin == 1

X2_train = X_train_proc.loc[train_mask]
y2_train = y_train.loc[train_mask]

X2_valid = X_valid_proc.loc[valid_mask]
y2_valid = y_valid.loc[valid_mask]

print("Stage 2 train shape:", X2_train.shape)
print("Stage 2 valid shape:", X2_valid.shape)


import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

lgb_stage2 = LGBMRegressor(
    boosting_type="gbdt",
    objective="regression",
    learning_rate=0.03,
    n_estimators=3000,
    num_leaves=63,
    max_depth=30,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=100,
    n_jobs=-1,
    importance_type="gain"
)

lgb_stage2.fit(
    X2_train,
    y2_train,
    eval_set=[(X2_valid, y2_valid)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
)

train_pred_2 = lgb_stage2.predict(X2_train)
valid_pred_2 = lgb_stage2.predict(X2_valid)

print("STAGE 2 – TRAIN METRICS")
print("MAE :", round(mean_absolute_error(y2_train, train_pred_2), 2))
print("RMSE:", round(root_mean_squared_error(y2_train, train_pred_2), 2))
print("R2  :", round(r2_score(y2_train, train_pred_2), 4))

print("\nSTAGE 2 – VALID METRICS")
print("MAE :", round(mean_absolute_error(y2_valid, valid_pred_2), 2))
print("RMSE:", round(root_mean_squared_error(y2_valid, valid_pred_2), 2))
print("R2  :", round(r2_score(y2_valid, valid_pred_2), 4))


feat_imp_stage2 = (
    pd.DataFrame({
        "feature": X2_train.columns,
        "importance": lgb_stage2.feature_importances_
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

feat_imp_stage2.head(20)




# Stage 1 probabilities
p_stage1_valid = lgb_stage1.predict_proba(X_valid_proc)[:, 1]

final_pred_valid = np.zeros(len(X_valid_proc))

mask = p_stage1_valid >= STAGE1_THRESHOLD
final_pred_valid[mask] = lgb_stage2.predict(X_valid_proc.loc[mask])

print("FINAL VALID METRICS (FULL POPULATION)")
print("MAE :", round(mean_absolute_error(y_valid, final_pred_valid), 2))
print("RMSE:", round(root_mean_squared_error(y_valid, final_pred_valid), 2))
print("R2  :", round(r2_score(y_valid, final_pred_valid), 4))
