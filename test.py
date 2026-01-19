cat_features = [c for c in df.columns if df[c].dtype.name in ["category", "object"]]
for c in cat_features:
    df[c] = df[c].astype("category")


TARGET_COL = "CURR_ACC"

df = df[df[TARGET_COL] > 100].copy()

X = df.drop(columns=[TARGET_COL])
y = np.log1p(df[TARGET_COL])


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


import lightgbm as lgb

reg = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

reg.fit(
    X_train,
    y_train,
    categorical_feature=cat_features,
    eval_set=[(X_val, y_val)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(50)]
)

from sklearn.metrics import mean_absolute_error

y_pred_log = reg.predict(X_val)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_val)

mae = mean_absolute_error(y_true, y_pred)
print(f"Final MAE: {mae:,.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_true, y_pred, alpha=0.3, s=10)
plt.plot([1, y_true.max()], [1, y_true.max()], "r--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True CURR_ACC")
plt.ylabel("Predicted CURR_ACC")
plt.title("True vs Predicted")
plt.show()
