import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

cat_cols = [c for c in X_train_clf.columns if X_train_clf[c].dtype.name in ("object", "category")]

X_train_clf = X_train_clf.copy()
X_val_clf = X_val_clf.copy()

for c in cat_cols:
    X_train_clf[c] = X_train_clf[c].astype("category")
    X_val_clf[c] = X_val_clf[c].astype("category")

neg = (y_train_clf == 0).sum()
pos = (y_train_clf == 1).sum()
scale_pos_weight = neg / pos

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

clf.fit(
    X_train_clf,
    y_train_clf,
    eval_set=[(X_val_clf, y_val_clf)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=True)]
)

y_pred_proba = clf.predict_proba(X_val_clf)[:, 1]

print("ROC-AUC:", roc_auc_score(y_val_clf, y_pred_proba))
print("PR-AUC:", average_precision_score(y_val_clf, y_pred_proba))

for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    y_pred_thr = (y_pred_proba >= thr).astype(int)
    print(f"\n{'='*50}")
    print(f"THRESHOLD = {thr}")
    print(confusion_matrix(y_val_clf, y_pred_thr))
    print(classification_report(y_val_clf, y_pred_thr, digits=4))