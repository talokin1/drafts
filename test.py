TARGET_COL = "CURR_ACC"

# categorical features
cat_features = [c for c in df.columns if df[c].dtype.name in ["object", "category"]]
for c in cat_features:
    df[c] = df[c].astype("category")

# target
y = df[TARGET_COL].clip(lower=0)
X = df.drop(columns=[TARGET_COL])

# stage-1 labels (ZERO vs POSITIVE)
y_cls = (y > 0).astype(int)

print("Class balance:")
print(y_cls.value_counts(normalize=True))

X_train, X_val, y_train, y_val, y_cls_train, y_cls_val = train_test_split(
    X, y, y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

assert y_cls_train.nunique() == 2
assert y_cls_val.nunique() == 2



clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    class_weight="balanced",
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

clf.fit(
    X_train,
    y_cls_train,
    categorical_feature=cat_features,
    eval_set=[(X_val, y_cls_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50)],
)
auc = roc_auc_score(y_cls_val, clf.predict_proba(X_val)[:, 1])
print(f"Stage-1 ROC-AUC: {auc:.4f}")



mask_train_pos = y_train > 0
mask_val_pos   = y_val > 0

X_train_reg = X_train[mask_train_pos]
X_val_reg   = X_val[mask_val_pos]

y_train_reg = np.log1p(y_train[mask_train_pos])
y_val_reg   = np.log1p(y_val[mask_val_pos])


reg = lgb.LGBMRegressor(
    objective="quantile",      # краще для heavy-tail
    alpha=0.5,
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
    X_train_reg,
    y_train_reg,
    categorical_feature=cat_features,
    eval_set=[(X_val_reg, y_val_reg)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(50)],
)

def predict_two_stage(X, clf, reg, threshold):
    probs = clf.predict_proba(X)[:, 1]
    preds = np.zeros(len(X))

    mask = probs > threshold
    if mask.sum() > 0:
        preds_log = reg.predict(X[mask])
        preds[mask] = np.expm1(preds_log)

    return preds

probs = clf.predict_proba(X_val)[:, 1]

best = None
for t in np.linspace(0.1, 0.9, 81):
    preds = predict_two_stage(X_val, clf, reg, threshold=t)
    mae = mean_absolute_error(y_val, preds)
    if best is None or mae < best[0]:
        best = (mae, t)

best_mae, best_t = best
print(f"Best threshold: {best_t:.2f}")
print(f"Best MAE: {best_mae:,.2f}")

y_pred = predict_two_stage(X_val, clf, reg, threshold=best_t)

mae = mean_absolute_error(y_val, y_pred)
r2  = r2_score(y_val, y_pred)

print("=" * 40)
print(f"FINAL MAE (грн): {mae:,.2f}")
print(f"FINAL R²:       {r2:.4f}")
print("=" * 40)


mask_plot = (y_val > 0) & (y_pred > 0)

plt.figure(figsize=(8,6))
plt.scatter(y_val[mask_plot], y_pred[mask_plot], alpha=0.3, s=10)
plt.plot([1, y_val.max()], [1, y_val.max()], "r--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True CURR_ACC")
plt.ylabel("Predicted CURR_ACC")
plt.title("Two-Stage Model: True vs Predicted")
plt.show()


# baseline 0
mae_zero = mean_absolute_error(y_val, np.zeros_like(y_val))

# baseline median (only for positive)
med = np.median(y_train[y_train > 0])
pred_med = np.zeros_like(y_val)
pred_med[y_val > 0] = med
mae_med = mean_absolute_error(y_val, pred_med)

print(f"Baseline ZERO MAE:   {mae_zero:,.2f}")
print(f"Baseline MEDIAN MAE: {mae_med:,.2f}")
