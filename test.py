for col in scale_cols:
    if col in df.columns:
        df[col + "_log"] = np.log1p(df[col].clip(lower=0))


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

for col in diff_cols:
    if col in df.columns:
        df[col + "_slog"] = signed_log1p(df[col])

df["NB_EMPL_log"] = np.log1p(df["NB_EMPL"])


drop_prev = [c for c in df.columns if c.endswith("_PREV")]
df.drop(columns=drop_prev, inplace=True, errors="ignore")

df.drop(
    columns=[c for c in scale_cols + diff_cols if c in df.columns],
    inplace=True,
    errors="ignore"
)

features = (
    [c for c in df.columns if c.endswith("_log")]
    + [c for c in df.columns if c.endswith("_slog")]
    + ratio_cols
    + ["NB_EMPL_log"]
    + cat_cols
)






from sklearn.model_selection import train_test_split

X = df[features]
y = y.loc[X.index]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from lightgbm import LGBMRegressor

model = LGBMRegressor(
    objective="regression_l1",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="mae",
    categorical_feature=cat_cols,
    verbose=100
)

from sklearn.metrics import mean_absolute_error

y_pred_log = model.predict(X_valid)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_valid)

mae = mean_absolute_error(y_true, y_pred)
mae

import matplotlib.pyplot as plt

plt.scatter(y_true, np.abs(y_true - y_pred), alpha=0.2)
plt.xlabel("True CURR_ACC")
plt.ylabel("Absolute Error")
plt.xscale("log")
plt.yscale("log")
plt.show()

