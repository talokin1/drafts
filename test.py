y_train_proba_stage1 = lgb_stage1.predict_proba(X_train_proc)[:, 1]

thresholds = [0.3, 0.5, 0.7]

print("STAGE 1 TRAIN F1 BY THRESHOLD")
for t in thresholds:
    y_train_pred = (y_train_proba_stage1 >= t).astype(int)
    f1 = f1_score(y_train_bin, y_train_pred)
    print(f"Threshold {t}: F1 = {f1:.4f}")
