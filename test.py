TARGET_NAME = "FX"
RANDOM_STATE = 42

df_base = df.copy()
df_base[TARGET_NAME] = pd.to_numeric(df_base[TARGET_NAME], errors="coerce").fillna(0)
df_base[TARGET_NAME] = df_base[TARGET_NAME].clip(lower=0)

y_clf_full = (df_base[TARGET_NAME] > 0).astype(int)

train_idx, val_idx = train_test_split(
    df_base.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_clf_full
)

df_train = df_base.loc[train_idx].copy()
df_val = df_base.loc[val_idx].copy()

# classification target
y_train_clf = (df_train[TARGET_NAME] > 0).astype(int)
y_val_clf = (df_val[TARGET_NAME] > 0).astype(int)

X_train_clf = df_train[final_features].copy()
X_val_clf = df_val[final_features].copy()

cat_cols = [c for c in X_train_clf.columns if X_train_clf[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train_clf.loc[:, c] = X_train_clf[c].astype("category")
    X_val_clf.loc[:, c] = X_val_clf[c].astype("category")
    





df_train_reg = df_train[df_train[TARGET_NAME] > 0].copy()
df_val_reg = df_val[df_val[TARGET_NAME] > 0].copy()

upper = df_train_reg[TARGET_NAME].quantile(0.99)

df_train_reg["FX_CAPPED"] = df_train_reg[TARGET_NAME].clip(upper=upper)
df_val_reg["FX_CAPPED"] = df_val_reg[TARGET_NAME].clip(upper=upper)

df_train_reg["FX_LOG"] = np.log1p(df_train_reg["FX_CAPPED"])
df_val_reg["FX_LOG"] = np.log1p(df_val_reg["FX_CAPPED"])

X_train_reg = df_train_reg[final_features].copy()
X_val_reg = df_val_reg[final_features].copy()

for c in cat_cols:
    if c in X_train_reg.columns:
        X_train_reg.loc[:, c] = X_train_reg[c].astype("category")
        X_val_reg.loc[:, c] = X_val_reg[c].astype("category")

y_train_reg = df_train_reg["FX_LOG"]
y_val_reg = df_val_reg["FX_LOG"]









# probabilities for all validation clients
p_fx = clf_binary.predict_proba(X_val_clf)[:, 1]

# conditional amount for all validation clients
X_val_all_reg = df_val[final_features].copy()

for c in cat_cols:
    X_val_all_reg.loc[:, c] = X_val_all_reg[c].astype("category")

pred_log_cond = reg.predict(X_val_all_reg)
pred_amount_cond = np.expm1(pred_log_cond)

# final expected FX potential
df_val_result = df_val[[TARGET_NAME]].copy()
df_val_result["P_FX"] = p_fx
df_val_result["FX_COND_PRED"] = pred_amount_cond
df_val_result["FX_EXPECTED"] = df_val_result["P_FX"] * df_val_result["FX_COND_PRED"]








from sklearn.metrics import mean_absolute_error, median_absolute_error

y_true_all = df_val_result[TARGET_NAME]
y_pred_all = df_val_result["FX_EXPECTED"]

print("MAE all:", mean_absolute_error(y_true_all, y_pred_all))
print("MedAE all:", median_absolute_error(y_true_all, y_pred_all))
print("True sum:", y_true_all.sum())
print("Pred sum:", y_pred_all.sum())
print("Pred / True:", y_pred_all.sum() / max(y_true_all.sum(), 1))