df_train_reg = df_train_proc[df_train[TARGET_NAME] > 0].copy()
df_val_reg   = df_val_proc[df_val[TARGET_NAME] > 0].copy()

# preprocess target (лог + кліпінг якщо є)
df_train_reg = preprocess_target(df_train_reg)
df_val_reg   = preprocess_target(df_val_reg)

X_train_reg = df_train_reg[final_features].copy()
X_val_reg   = df_val_reg[final_features].copy()

y_train_reg = df_train_reg[TARGET_NAME]
y_val_reg   = df_val_reg[TARGET_NAME]

cat_cols = [c for c in X_train_reg.columns if X_train_reg[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train_reg[c] = X_train_reg[c].astype("category")
    X_val_reg[c] = X_val_reg[c].astype("category")


reg = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train_reg,
    y_train_reg,
    eval_set=[(X_val_reg, y_val_reg)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(200, verbose=True)]
)


y_pred_log = reg.predict(X_val_reg)

y_val = np.expm1(y_val_reg)
y_pred = np.expm1(y_pred_log)

validation_results_reg = pd.DataFrame({
    "IDENTIFYCODE": df_val_reg.index,
    "True": y_val,
    "Predicted": y_pred
})


mae_log = mean_absolute_error(y_val_reg, y_pred_log)
r2_log = r2_score(y_val_reg, y_pred_log)
medae_log = median_absolute_error(y_val_reg, y_pred_log)

mae = mean_absolute_error(y_val, y_pred)
medae = median_absolute_error(y_val, y_pred)

eps = 1e-9
mape = np.mean(np.abs(y_val - y_pred) / np.maximum(np.abs(y_val), eps))
smape = np.mean(2.0 * np.abs(y_val - y_pred) / np.maximum(np.abs(y_val) + np.abs(y_pred), eps))


print("=" * 60)
print("REGRESSION METRICS (LOG SPACE)")
print(f"MAE_log   : {mae_log:.5f}")
print(f"MedAE_log : {medae_log:.5f}")
print(f"R2_log    : {r2_log:.5f}")

print("=" * 60)
print("REGRESSION METRICS (ORIGINAL SCALE)")
print(f"MAE   : {mae:.2f}")
print(f"MedAE : {medae:.2f}")
print(f"MAPE  : {mape:.4f}")
print(f"SMAPE : {smape:.4f}")
print("=" * 60)