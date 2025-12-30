target_like = [c for c in X_train_proc.columns if "income" in c.lower()]
target_like[:50]


bad = set(X_train_proc.columns) & set(y_train.to_frame().columns)
print("Overlap X vs y:", bad)



import numpy as np
import pandas as pd

num_cols = X_train_proc.select_dtypes(include=[np.number]).columns
corr_bin = X_train_proc[num_cols].corrwith(y_train_bin, method="spearman").sort_values(ascending=False)

corr_bin.head(30), corr_bin.tail(30)



top_feat = corr_bin.abs().idxmax()
print("Top feature:", top_feat, "corr:", corr_bin[top_feat])

display(X_train_proc.groupby(y_train_bin)[top_feat].describe())




from sklearn.metrics import f1_score, roc_auc_score

p_valid = lgb_stage1.predict_proba(X_valid_proc)[:, 1]
pred_valid = (p_valid >= 0.5).astype(int)

print("STAGE1 VALID F1:", round(f1_score(y_valid_bin, pred_valid), 4))
print("STAGE1 VALID AUC:", round(roc_auc_score(y_valid_bin, p_valid), 4))



from lightgbm import LGBMClassifier
import lightgbm as lgb

lgb_stage1 = LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    class_weight="balanced",
    learning_rate=0.03,
    n_estimators=3000,

    # СИЛЬНЕ спрощення
    max_depth=6,
    num_leaves=31,
    min_child_samples=200,

    # Регуляризація
    reg_alpha=1.0,
    reg_lambda=5.0,

    # Рандомізація
    subsample=0.8,
    colsample_bytree=0.8,
    subsample_freq=1,

    random_state=100,
    n_jobs=-1
)

lgb_stage1.fit(
    X_train_proc, y_train_bin,
    eval_set=[(X_valid_proc, y_valid_bin)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=True)]
)
