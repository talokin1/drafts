X_train_clf, X_val_clf, y_train_clf, y_val_clf
X_train_reg, X_val_reg, y_train_reg, y_val_reg

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report

# категорії
cat_cols = [c for c in X_train_clf.columns if X_train_clf[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train_clf[c] = X_train_clf[c].astype("category")
    X_val_clf[c] = X_val_clf[c].astype("category")

clf_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

clf_model.fit(
    X_train_clf, y_train_clf,
    eval_set=[(X_val_clf, y_val_clf)],
    callbacks=[lgb.early_stopping(50)]
)

# --- метрики
probs_clf = clf_model.predict_proba(X_val_clf)[:, 1]

threshold = 0.3
preds_clf = (probs_clf > threshold).astype(int)

print("\n=== CLASSIFICATION ===")
print("ROC-AUC:", roc_auc_score(y_val_clf, probs_clf))
print(classification_report(y_val_clf, preds_clf))


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# --- bins
bins = [-1, 350, 550, 750, 925, 1100, 2000, 4000, np.inf]

# --- target bins
y_train_binned = pd.cut(y_train_reg, bins=bins, labels=False)
y_val_binned   = pd.cut(y_val_reg, bins=bins, labels=False)

# --- категорії
for c in cat_cols:
    if c in X_train_reg.columns:
        X_train_reg[c] = X_train_reg[c].astype("category")
        X_val_reg[c] = X_val_reg[c].astype("category")

# --- bucket medians
bucket_medians = y_train_reg.groupby(y_train_binned).median().to_dict()

print("\nBucket medians:")
print(bucket_medians)

# --- model
reg_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(bins)-1,
    n_estimators=1500,
    learning_rate=0.03,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

reg_model.fit(
    X_train_reg, y_train_binned,
    eval_set=[(X_val_reg, y_val_binned)],
    callbacks=[lgb.early_stopping(50)]
)

# --- prediction (E[Y])
probs_reg = reg_model.predict_proba(X_val_reg)

y_pred_reg = np.zeros(len(X_val_reg))
for i in range(len(bins)-1):
    y_pred_reg += probs_reg[:, i] * bucket_medians[i]

mae_reg = mean_absolute_error(y_val_reg, y_pred_reg)

print("\n=== REGRESSION ===")
print("MAE (E[Y]):", mae_reg)


# --- classification
probs_all = clf_model.predict_proba(X_val_clf)[:, 1]
mask = probs_all > threshold

# --- фінальний предикт
y_pred_final = np.zeros(len(X_val_clf))

# --- регресія тільки для тих, кого clf пропустив
if mask.sum() > 0:
    X_val_selected = X_val_clf[mask]

    probs_selected = reg_model.predict_proba(X_val_selected)

    y_pred_selected = np.zeros(len(X_val_selected))
    for i in range(len(bins)-1):
        y_pred_selected += probs_selected[:, i] * bucket_medians[i]

    y_pred_final[mask] = y_pred_selected


from sklearn.metrics import mean_absolute_error

mae_final = mean_absolute_error(
    y_val_clf * y_val_reg.reindex(X_val_clf.index, fill_value=0),  # true values
    y_pred_final
)

print("\n=== FINAL PIPELINE ===")
print("FINAL MAE:", mae_final)

