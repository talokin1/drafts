cat_cols = [c for c in X_train_clf.columns if X_train_clf[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train_clf[c] = X_train_clf[c].astype("category")
    X_val_clf[c]   = X_val_clf[c].astype("category")


clf_model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)

clf_model.fit(
    X_train_clf,
    y_train_clf,
    eval_set=[(X_val_clf, y_val_clf)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

y_val_proba = clf_model.predict_proba(X_val_clf)[:, 1]

print("ROC-AUC:", roc_auc_score(y_val_clf, y_val_proba))
print(classification_report(y_val_clf, (y_val_proba > 0.5).astype(int)))











for c in cat_cols:
    X_train_reg[c] = X_train_reg[c].astype("category")
    X_val_reg[c]   = X_val_reg[c].astype("category")

## my modelssss












proba_income = clf_model.predict_proba(X)[:, 1]
probs_multi = reg_model.predict_proba(X)  # або як у тебе зроблено
expected_value = np.sum(probs_multi * bucket_values, axis=1)

final_pred = proba_income * expected_value

