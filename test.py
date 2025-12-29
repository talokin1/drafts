y_train_bin = (y_train > 0).astype(int)
y_valid_bin = (y_valid > 0).astype(int)


from lightgbm import LGBMClassifier
import lightgbm as lgb
lgb_clf = LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    learning_rate=0.03,
    n_estimators=2000,
    num_leaves=41,
    max_depth=20,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    class_weight="balanced",
    random_state=100,
    n_jobs=-1
)

lgb_clf.fit(
    X_train_proc,
    y_train_bin,
    eval_set=[(X_valid_proc, y_valid_bin)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)

from sklearn.metrics import roc_auc_score, f1_score

p_valid_nonzero = lgb_clf.predict_proba(X_valid_proc)[:, 1]

auc_stage1 = roc_auc_score(y_valid_bin, p_valid_nonzero)
f1_stage1 = f1_score(y_valid_bin, (p_valid_nonzero >= 0.5).astype(int))

print("STAGE 1 (Classification)")
print("AUC:", round(auc_stage1, 4))
print("F1 :", round(f1_stage1, 4))







mask_pos_train = y_train > 0

X_train_pos = X_train_proc[mask_pos_train]
y_train_pos = y_train[mask_pos_train]
from lightgbm import LGBMRegressor

lgb_reg_pos = LGBMRegressor(
    boosting_type="gbdt",
    objective="regression",
    learning_rate=0.02,
    n_estimators=3000,
    num_leaves=41,
    max_depth=30,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=100,
    n_jobs=-1,
    importance_type="gain"
)
lgb_reg_pos.fit(
    X_train_pos,
    y_train_pos,
    eval_set=[(X_valid_proc[y_valid > 0], y_valid[y_valid > 0])],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)





# P(y > 0)
p_nonzero_valid = lgb_clf.predict_proba(X_valid_proc)[:, 1]

# E[y | y > 0]
y_pos_pred_valid = lgb_reg_pos.predict(X_valid_proc)

# Final prediction
y_valid_two_stage = p_nonzero_valid * y_pos_pred_valid




from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
mae_2s = mean_absolute_error(y_valid, y_valid_two_stage)
rmse_2s = root_mean_squared_error(y_valid, y_valid_two_stage, squared=False)
r2_2s = r2_score(y_valid, y_valid_two_stage)

print("TWO-STAGE MODEL (VALID)")
print("MAE :", round(mae_2s, 2))
print("RMSE:", round(rmse_2s, 2))
print("R2  :", round(r2_2s, 4))

X_valid_two_stage = pd.concat(
    [
        X_valid.reset_index(drop=True),
        y_valid.reset_index(drop=True),
        pd.Series(y_valid_two_stage, name="PRED_TWO_STAGE")
    ],
    axis=1
)

X_valid_two_stage.to_csv(
    f"x_valid_two_stage_result_{month_next}.csv",
    index=False
)
