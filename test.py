from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import lightgbm as lgb


# =========================
# 1. DATA FOR REGRESSION
# =========================
df_reg = df[df[TARGET_NAME] > 0].copy()
df_reg = preprocess_target(df_reg)

y_raw = df_reg[TARGET_NAME]
X_ = df_reg[final_features]


# =========================
# 2. TRAIN / VAL SPLIT
# =========================
X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    X_, y_raw,
    test_size=0.2,
    random_state=RANDOM_STATE
)


# =========================
# 3. BUCKETS
# =========================
quantiles = np.linspace(0, 1, 9)
bins = np.quantile(y_train_raw, quantiles)

bins[0] = -np.inf
bins[-1] = np.inf
bins = np.unique(bins)

y_train_bins = pd.cut(y_train_raw, bins=bins, labels=False)
y_val_bins   = pd.cut(y_val_raw, bins=bins, labels=False)


# =========================
# 4. BUCKET MEDIANS
# =========================
bucket_medians = (
    pd.DataFrame({"bin": y_train_bins, "y": y_train_raw})
    .groupby("bin")["y"]
    .median()
)


# =========================
# 5. MULTICLASS MODEL
# =========================
clf_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(bins)-1,
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf_model.fit(
    X_train,
    y_train_bins,
    eval_set=[(X_val, y_val_bins)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)


# =========================
# 6. PREDICT MULTICLASS
# =========================
probs = clf_model.predict_proba(X_val)

y_pred_expected = np.zeros(len(X_val))

for i in range(probs.shape[1]):
    if i in bucket_medians.index:
        y_pred_expected += probs[:, i] * bucket_medians.loc[i]


# =========================
# 7. BACK TO ORIGINAL SCALE
# =========================
y_val_orig = np.expm1(y_val_raw)
y_pred_orig = np.expm1(y_pred_expected)


# =========================
# 8. METRICS (REGRESSION ONLY)
# =========================
mae_multiclass = mean_absolute_error(y_val_orig, y_pred_orig)

print("\n=== MULTICLASS REGRESSION ===")
print("MAE (E[y]):", mae_multiclass)


# =========================
# 9. CLASSIFIER METRICS
# =========================
# (припускаємо, що clf_binary вже навчений)

X_val_full = df.loc[X_val.index, final_features]
y_val_clf = (df.loc[X_val.index, TARGET_NAME] > 0).astype(int)

probs_clf = clf_binary.predict_proba(X_val_full)[:, 1]

roc = roc_auc_score(y_val_clf, probs_clf)

print("\n=== CLASSIFICATION ===")
print("ROC-AUC:", roc)


# =========================
# 10. FINAL PREDICTION
# =========================
p_income = probs_clf
y_expected = y_pred_orig

final_prediction = p_income * y_expected


# =========================
# 11. FINAL METRIC (ГОЛОВНА)
# =========================
y_val_true_full = df.loc[X_val.index, TARGET_NAME]

mae_final = mean_absolute_error(y_val_true_full, final_prediction)

print("\n=== FINAL MODEL ===")
print("MAE (P * E[y]):", mae_final)