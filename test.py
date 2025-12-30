import lightgbm as lgb
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)


lgb_stage1 = LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    class_weight="balanced",

    learning_rate=0.02,
    n_estimators=3000,
    num_leaves=41,
    max_depth=30,

    min_child_samples=50,    
    min_child_weight=1.0,
    min_split_gain=0.0,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=100,
    n_jobs=-1,
    importance_type="gain"
)
lgb_stage1.fit(
    X_train_proc,
    y_train_bin,
    eval_set=[(X_valid_proc, y_valid_bin)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)


p_train = lgb_stage1.predict_proba(X_train_proc)[:, 1]
y_train_pred = (p_train >= 0.5).astype(int)

print("STAGE 1 — TRAIN METRICS")
print("F1:", round(f1_score(y_train_bin, y_train_pred), 4))
print("Precision:", round(precision_score(y_train_bin, y_train_pred), 4))
print("Recall:", round(recall_score(y_train_bin, y_train_pred), 4))




p_valid = lgb_stage1.predict_proba(X_valid_proc)[:, 1]
y_valid_pred = (p_valid >= 0.5).astype(int)

print("\nSTAGE 1 — VALID METRICS")
print("F1:", round(f1_score(y_valid_bin, y_valid_pred), 4))
print("AUC:", round(roc_auc_score(y_valid_bin, p_valid), 4))
print("Precision:", round(precision_score(y_valid_bin, y_valid_pred), 4))
print("Recall:", round(recall_score(y_valid_bin, y_valid_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_valid_bin, y_valid_pred))



thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\nSTAGE 1 — VALID F1 BY THRESHOLD")
for t in thresholds:
    y_pred_t = (p_valid >= t).astype(int)
    f1 = f1_score(y_valid_bin, y_pred_t)
    print(f"Threshold {t}: F1 = {f1:.4f}")
