ZERO_THRESHOLD = 1.0

BUCKET_LABELS = [
    "zero",
    "1-100",
    "100-500",
    "500-1.5k",
    "1.5k-5k",
    "5k-15k",
    "15k-50k",
    "50k-150k",
    "150k+"
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
    buckets[(y_clean > 50_000) & (y_clean <= 150_000)] = 7
    buckets[y_clean > 150_000] = 8

    buckets = buckets.fillna(0).astype(int)

    bucket_names = pd.Series(
        [BUCKET_LABELS[i] for i in buckets],
        index=y_clean.index
    )

    return buckets, bucket_names


y_bucket, y_bucket_name = make_target_buckets(y_clean)



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
        else:
            bucket_values[bucket_id] = vals.median()

    return bucket_values


P_ZERO_THRESHOLD = 0.65

P_ZERO_THRESHOLD = 0.65

train_p_zero = train_proba[:, 0]
val_p_zero = val_proba[:, 0]

train_expected_raw[train_p_zero >= P_ZERO_THRESHOLD] = 0
val_expected_raw[val_p_zero >= P_ZERO_THRESHOLD] = 0


validation_results["P_ZERO"] = val_proba[:, 0]
validation_results["IS_PRED_ZERO"] = (validation_results["P_ZERO"] >= P_ZERO_THRESHOLD).astype(int)

validation_results["P_ZERO"] = validation_results["P_ZERO"].round(4)


validation_results_view = validation_results[
    [
        ID_COL,
        SEGMENT_COL,
        "True_Value",
        "True_Bucket",
        "Pred_Bucket",
        "P_ZERO",
        "IS_PRED_ZERO",
        "Expected_Raw",
        "Expected_Calibrated",
        "Predicted",
        "Error",
        "Abs_Error",
        "Ratio"
    ]
].copy()







model_artifacts = {
    "model_type": "bucket_expected_value_with_zero_bucket",

    "bucket_model": bucket_model,

    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "cat_values": cat_values,

    "bucket_labels": BUCKET_LABELS,
    "n_buckets": N_BUCKETS,
    "bucket_values": bucket_values,

    "p_zero_threshold": P_ZERO_THRESHOLD,

    "segment_col": SEGMENT_COL,

    "segment_calibration": segment_calibration,
    "global_calibration_factor": global_calibration_factor,

    "caps_by_segment": caps_by_segment,
    "global_cap": global_cap
}




P_ZERO_THRESHOLD = artifacts.get("p_zero_threshold", 0.65)

p_zero = proba[:, 0]
expected_raw[p_zero >= P_ZERO_THRESHOLD] = 0

result["P_ZERO"] = p_zero
result["IS_PRED_ZERO"] = (p_zero >= P_ZERO_THRESHOLD).astype(int)