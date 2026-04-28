ZERO_THRESHOLD = 1.0

BUCKET_LABELS = [
    "zero",
    "1-100",
    "100-500",
    "500-1.5k",
    "1.5k-5k",
    "5k-15k",
    "15k-50k",
    "50k+"
]

N_BUCKETS = len(BUCKET_LABELS)



def make_target_buckets(y):
    y_clean = pd.Series(y).clip(lower=0)

    buckets = pd.Series(index=y_clean.index, dtype=int)

    buckets[y_clean <= 1] = 0
    buckets[(y_clean > 1) & (y_clean <= 100)] = 1
    buckets[(y_clean > 100) & (y_clean <= 500)] = 2
    buckets[(y_clean > 500) & (y_clean <= 1_500)] = 3
    buckets[(y_clean > 1_500) & (y_clean <= 5_000)] = 4
    buckets[(y_clean > 5_000) & (y_clean <= 15_000)] = 5
    buckets[(y_clean > 15_000) & (y_clean <= 50_000)] = 6
    buckets[y_clean > 50_000] = 7

    buckets = buckets.fillna(0).astype(int)

    bucket_names = pd.Series(
        [BUCKET_LABELS[i] for i in buckets],
        index=y_clean.index
    )

    return buckets, bucket_names


y_bucket, y_bucket_name = make_target_buckets(y_clean)

print("Bucket distribution:")
bucket_dist = (
    pd.DataFrame({
        "bucket_id": y_bucket,
        "bucket_name": y_bucket_name
    })
    .groupby(["bucket_id", "bucket_name"])
    .size()
    .reset_index(name="count")
)

bucket_dist["share"] = bucket_dist["count"] / bucket_dist["count"].sum()
display(bucket_dist)



class_weights = build_bucket_class_weights(y_train_bucket)

bucket_weight = pd.Series(y_train_bucket).map(class_weights).values
segment_weight = build_segment_weights(X_train).values

sample_weight = np.sqrt(bucket_weight) * np.sqrt(segment_weight)

print("Class weights:")
print(class_weights)

print("Sample weight describe:")
print(pd.Series(sample_weight).describe())



bucket_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=N_BUCKETS,

    n_estimators=1200,
    learning_rate=0.04,

    num_leaves=15,
    max_depth=5,
    min_child_samples=10,

    subsample=0.9,
    colsample_bytree=0.9,

    reg_alpha=0.1,
    reg_lambda=0.5,

    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

print("Training Bucket Expected Value Model...")

bucket_model.fit(
    X_train,
    y_train_bucket,
    sample_weight=sample_weight,
    eval_set=[(X_val, y_val_bucket)],
    eval_metric="multi_logloss",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

print("Best iteration:", bucket_model.best_iteration_)




def build_bucket_values(y_train, y_train_bucket, bucket_labels):
    tmp = pd.DataFrame({
        "target": pd.Series(y_train).clip(lower=0),
        "bucket": y_train_bucket
    })

    bucket_values = {}

    for bucket_id, bucket_name in enumerate(bucket_labels):
        vals = tmp.loc[tmp["bucket"] == bucket_id, "target"]

        if bucket_id == 0:
            bucket_values[bucket_id] = 0.0
            continue

        if len(vals) == 0:
            bucket_values[bucket_id] = 0.0
            continue

        if bucket_name == "50k+":
            # хвіст: не mean, бо mean може рознести прогноз
            # не median, бо може занизити топ
            bucket_values[bucket_id] = vals.quantile(0.75)
        else:
            bucket_values[bucket_id] = vals.median()

    return bucket_values


bucket_values = build_bucket_values(
    y_train=y_train_raw,
    y_train_bucket=y_train_bucket,
    bucket_labels=BUCKET_LABELS
)

bucket_value_table = pd.DataFrame({
    "bucket_id": list(bucket_values.keys()),
    "bucket_name": [BUCKET_LABELS[i] for i in bucket_values.keys()],
    "bucket_value": list(bucket_values.values())
})

display(bucket_value_table)



P_ZERO_THRESHOLD = 0.65
train_expected_raw = expected_value_from_proba(train_proba, bucket_values)
val_expected_raw = expected_value_from_proba(val_proba, bucket_values)

train_p_zero = train_proba[:, 0]
val_p_zero = val_proba[:, 0]

train_expected_raw[train_p_zero >= P_ZERO_THRESHOLD] = 0
val_expected_raw[val_p_zero >= P_ZERO_THRESHOLD] = 0






validation_results["P_ZERO"] = val_proba[:, 0]
validation_results["IS_PRED_ZERO"] = (validation_results["P_ZERO"] >= P_ZERO_THRESHOLD).astype(int)



"p_zero_threshold": P_ZERO_THRESHOLD,


