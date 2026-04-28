print("Навчання Stage 2: Regressor...")
q = y_train_raw.quantile(0.97)

mask_train_reg = (
    (y_train_raw >= THRESHOLD) & 
    (y_train_raw > 0.5) & 
    (y_train_raw < q)
).values

X_train_reg = X_train[mask_train_reg].copy()
y_train_reg_log = np.log1p(y_train_raw[mask_train_reg]) # Або без логарифма, якщо перейшов на MAE



mask_val_reg = (
    (y_val_raw >= THRESHOLD) & 
    (y_val_raw > 0.5) & 
    (y_val_raw < q)
).values

X_val_reg = X_val[mask_val_reg].copy()
y_val_reg_log = np.log1p(y_val_raw[mask_val_reg])

reg = lgb.LGBMRegressor(
    objective="regression", 
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    categorical_feature=cat_cols,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)