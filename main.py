print("Навчання Stage 2: Conservative Regressor...")

# ============================================================
# ACTIVE MASKS
# ============================================================

mask_train_active = (y_train_raw >= THRESHOLD).values
mask_val_active = (y_val_raw >= THRESHOLD).values

X_train_reg = X_train[mask_train_active].copy()
X_val_reg = X_val[mask_val_active].copy()

y_train_reg_raw = y_train_raw[mask_train_active].copy()
y_val_reg_raw = y_val_raw[mask_val_active].copy()


# ============================================================
# TARGET CAP — тільки по train, без leakage
# ============================================================

REG_TARGET_CAP_Q = 0.97

target_cap = np.quantile(y_train_reg_raw, REG_TARGET_CAP_Q)

print(f"Regression target cap q={REG_TARGET_CAP_Q}: {target_cap:,.2f}")

y_train_reg_capped = np.clip(y_train_reg_raw, 0, target_cap)
y_val_reg_capped = np.clip(y_val_reg_raw, 0, target_cap)

y_train_reg_log = np.log1p(y_train_reg_capped)
y_val_reg_log = np.log1p(y_val_reg_capped)


# ============================================================
# SAMPLE WEIGHTS
# Менше ваги великим клієнтам, щоб модель не вчилася тільки на хвості
# ============================================================

sample_weight_reg = 1 / np.sqrt(1 + y_train_reg_capped)
sample_weight_reg = sample_weight_reg / sample_weight_reg.mean()


# ============================================================
# CONSERVATIVE REGRESSOR
# ============================================================

reg = lgb.LGBMRegressor(
    objective="regression_l1",
    metric="mae",

    n_estimators=3000,
    learning_rate=0.02,

    num_leaves=15,
    max_depth=4,
    min_child_samples=100,

    subsample=0.75,
    colsample_bytree=0.75,

    reg_alpha=5.0,
    reg_lambda=20.0,

    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train_reg,
    y_train_reg_log,
    sample_weight=sample_weight_reg,
    eval_set=[(X_val_reg, y_val_reg_log)],
    eval_metric="mae",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False)
    ]
)


# ============================================================
# REGRESSION DIAGNOSTICS ON ACTIVE CLIENTS
# ============================================================

train_reg_pred_log = reg.predict(X_train_reg)
val_reg_pred_log = reg.predict(X_val_reg)

train_reg_pred = np.expm1(train_reg_pred_log)
val_reg_pred = np.expm1(val_reg_pred_log)

train_reg_pred = np.clip(train_reg_pred, 0, target_cap)
val_reg_pred = np.clip(val_reg_pred, 0, target_cap)

print("=" * 70)
print("[Stage 2: Conservative Regressor, ACTIVE ONLY]")
print("=" * 70)

print("[LOG SPACE]")
print(
    f"MAE    | Train: {mean_absolute_error(np.log1p(y_train_reg_raw), np.log1p(train_reg_pred)):.4f} "
    f"| Val: {mean_absolute_error(np.log1p(y_val_reg_raw), np.log1p(val_reg_pred)):.4f}"
)
print(
    f"MedAE  | Train: {median_absolute_error(np.log1p(y_train_reg_raw), np.log1p(train_reg_pred)):.4f} "
    f"| Val: {median_absolute_error(np.log1p(y_val_reg_raw), np.log1p(val_reg_pred)):.4f}"
)
print(
    f"R2     | Train: {r2_score(np.log1p(y_train_reg_raw), np.log1p(train_reg_pred)):.4f} "
    f"| Val: {r2_score(np.log1p(y_val_reg_raw), np.log1p(val_reg_pred)):.4f}"
)

print("-" * 70)
print("[ORIGINAL SPACE]")
print(
    f"MAE    | Train: {mean_absolute_error(y_train_reg_raw, train_reg_pred):,.2f} "
    f"| Val: {mean_absolute_error(y_val_reg_raw, val_reg_pred):,.2f}"
)
print(
    f"MedAE  | Train: {median_absolute_error(y_train_reg_raw, train_reg_pred):,.2f} "
    f"| Val: {median_absolute_error(y_val_reg_raw, val_reg_pred):,.2f}"
)
print(
    f"R2     | Train: {r2_score(y_train_reg_raw, train_reg_pred):.4f} "
    f"| Val: {r2_score(y_val_reg_raw, val_reg_pred):.4f}"
)

print("-" * 70)
print(f"Real active total train: {y_train_reg_raw.sum():,.2f}")
print(f"Pred active total train: {train_reg_pred.sum():,.2f}")
print(f"Ratio train            : {train_reg_pred.sum() / y_train_reg_raw.sum():.4f}")

print(f"Real active total val  : {y_val_reg_raw.sum():,.2f}")
print(f"Pred active total val  : {val_reg_pred.sum():,.2f}")
print(f"Ratio val              : {val_reg_pred.sum() / y_val_reg_raw.sum():.4f}")
print("=" * 70)