import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score


TARGET_COL = "CURR_ACC"

cat_features = [c for c in df.columns if df[c].dtype.name in ["category", "object"]]
for c in cat_features:
    df[c] = df[c].astype("category")


X_train, X_val, y_train, y_val, y_cls_train, y_cls_val = train_test_split(
    X, y, y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)


clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
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
    callbacks=[lgb.early_stopping(50)]
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
    objective="regression_l1",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=200,
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
    callbacks=[lgb.early_stopping(50)]
)


def predict_two_stage(X, clf, reg, threshold=0.5):
    probs = clf.predict_proba(X)[:, 1]
    preds = np.zeros(len(X))

    mask = probs > threshold
    if mask.sum() > 0:
        preds_log = reg.predict(X[mask])
        preds[mask] = np.expm1(preds_log)

    return preds


y_pred = predict_two_stage(X_val, clf, reg, threshold=0.5)

mae = mean_absolute_error(y_val, y_pred)
r2  = r2_score(y_val, y_pred)

print("=" * 40)
print(f"Final MAE (грн): {mae:,.2f}")
print(f"Final R² (k2):   {r2:.4f}")
print("=" * 40)

import matplotlib.pyplot as plt

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
