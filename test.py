# ============================================================
# STAGE 2: ACTIVE CLIENT BUCKET MODEL - STABLE VERSION
# ============================================================

BUCKET_QUANTILES = [0.00, 0.25, 0.50, 0.70, 0.85, 0.95, 1.00]

bucket_edges = make_bucket_edges(y_train_active, BUCKET_QUANTILES)
n_buckets = len(bucket_edges) - 1

y_train_bucket = assign_buckets(y_train_active, bucket_edges)
y_val_bucket = assign_buckets(y_val_active, bucket_edges)
y_test_bucket = assign_buckets(y_test_active, bucket_edges)

bucket_values = calculate_bucket_values(
    y_train_active=y_train_active,
    y_train_bucket=y_train_bucket,
    n_buckets=n_buckets
)

bucket_table = pd.DataFrame({
    "bucket": np.arange(n_buckets),
    "left_edge": bucket_edges[:-1],
    "right_edge": bucket_edges[1:],
    "bucket_value": bucket_values
})

display(bucket_table)

bucket_clf = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=n_buckets,

    n_estimators=1200,
    learning_rate=0.03,

    num_leaves=7,
    max_depth=3,
    min_child_samples=80,

    subsample=0.8,
    colsample_bytree=0.7,

    reg_alpha=3.0,
    reg_lambda=15.0,

    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

bucket_clf.fit(
    X_train_active,
    y_train_bucket,
    eval_set=[(X_val_active, y_val_bucket)],
    eval_metric="multi_logloss",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False)
    ]
)

bucket_pred_val_active = bucket_clf.predict(X_val_active)
bucket_pred_test_active = bucket_clf.predict(X_test_active)

print("Bucket classification report, VAL active clients:")
print(classification_report(y_val_bucket, bucket_pred_val_active, zero_division=0))

print("Bucket classification report, TEST active clients:")
print(classification_report(y_test_bucket, bucket_pred_test_active, zero_division=0))