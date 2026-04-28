df_model = X.copy()
y_clean = pd.Series(y, index=X.index).clip(lower=0)

y_binary = (y_clean > ACTIVE_THRESHOLD).astype(int)

print("Target describe:")
print(y_clean.describe())

print("\nActive distribution:")
print(y_binary.value_counts())
print(y_binary.value_counts(normalize=True))


X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    df_model,
    y_clean,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_binary
)

y_train_clf = (y_train_raw > ACTIVE_THRESHOLD).astype(int)
y_val_clf = (y_val_raw > ACTIVE_THRESHOLD).astype(int)

X_train, X_val, cat_cols, cat_values = prepare_categorical_train_valid(
    X_train,
    X_val
)

feature_cols = X_train.columns.tolist()

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("cat_cols:", cat_cols)










df_model = X.copy()
y_clean = pd.Series(y, index=X.index).clip(lower=0)

y_binary = (y_clean > ACTIVE_THRESHOLD).astype(int)

print("Target describe:")
print(y_clean.describe())

print("\nActive distribution:")
print(y_binary.value_counts())
print(y_binary.value_counts(normalize=True))


def build_segment_weights(X_part):
    weights = pd.Series(1.0, index=X_part.index, dtype=float)

    if SEGMENT_COL in X_part.columns:
        seg = X_part[SEGMENT_COL].astype(str)

        weights.loc[seg.eq("MICRO")] = 1.0
        weights.loc[seg.eq("SMALL")] = 1.2
        weights.loc[seg.eq("MEDIUM")] = 1.5
        weights.loc[seg.eq("LARGE")] = 2.5

    return weights

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


print("Training Stage 1: classifier...")

clf.fit(
    X_train,
    y_train_clf,
    sample_weight=clf_sample_weight,
    eval_set=[(X_val, y_val_clf)],
    eval_metric="auc",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

train_p_active = clf.predict_proba(X_train)[:, 1]
val_p_active = clf.predict_proba(X_val)[:, 1]

val_class_pred = (val_p_active >= CLASSIFICATION_THRESHOLD).astype(int)

print("=" * 80)
print("[Stage 1: Classifier]")
print(f"ROC-AUC Train: {roc_auc_score(y_train_clf, train_p_active):.4f}")
print(f"ROC-AUC Val  : {roc_auc_score(y_val_clf, val_p_active):.4f}")
print(f"PR-AUC Val   : {average_precision_score(y_val_clf, val_p_active):.4f}")
print("-" * 80)
print(classification_report(y_val_clf, val_class_pred))
print(confusion_matrix(y_val_clf, val_class_pred))
print("=" * 80)