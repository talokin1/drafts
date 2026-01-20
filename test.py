TARGET_COL = "CURR_ACC"

# categorical features
cat_features = [c for c in df.columns if df[c].dtype.name in ["object", "category"]]
for c in cat_features:
    df[c] = df[c].astype("category")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


good_features, df_imp = auto_select_features(
    X_train, y_train,
    X_val, y_val,
    cat_features=cat_features,
    task="regression",
    threshold=0.001
)

X_train = X_train[good_features]
X_val   = X_val[good_features]


reg = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=128,
    min_child_samples=100,
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
    callbacks=[lgb.early_stopping(50)],
)


y_pred = reg.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
r2  = r2_score(y_val, y_pred)

print("=" * 40)
print(f"FINAL MAE: {mae:,.2f}")
print(f"FINAL R² : {r2:.4f}")
print("=" * 40)

mae_zero = mean_absolute_error(y_val, np.zeros_like(y_val))

med = np.median(y_train)
pred_med = np.full_like(y_val, med)
mae_med = mean_absolute_error(y_val, pred_med)

print(f"Baseline ZERO MAE  : {mae_zero:,.2f}")
print(f"Baseline MEDIAN MAE: {mae_med:,.2f}")

plt.figure(figsize=(8,6))
plt.scatter(y_val, y_pred, alpha=0.3, s=10)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
plt.xlabel("True CURR_ACC (log)")
plt.ylabel("Predicted CURR_ACC (log)")
plt.title("LGBM Regression: True vs Predicted")
plt.show()
