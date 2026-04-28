mask_train_reg = (y_train_raw > ACTIVE_THRESHOLD).values
mask_val_reg = (y_val_raw > ACTIVE_THRESHOLD).values

X_train_reg = X_train.loc[mask_train_reg].copy()
X_val_reg = X_val.loc[mask_val_reg].copy()

y_train_reg_raw = y_train_raw.loc[mask_train_reg].copy()
y_val_reg_raw = y_val_raw.loc[mask_val_reg].copy()

y_train_reg_log = np.log1p(y_train_reg_raw)
y_val_reg_log = np.log1p(y_val_reg_raw)

print("Regression train:", X_train_reg.shape)
print("Regression val:", X_val_reg.shape)
print(y_train_reg_raw.describe())


reg_sample_weight = build_segment_weights(X_train_reg)

reg_sample_weight = reg_sample_weight * (
    1.0 + np.log1p(y_train_reg_raw) / np.log1p(y_train_reg_raw).max()
)

reg_sample_weight = build_segment_weights(X_train_reg)

reg_sample_weight = reg_sample_weight * (
    1.0 + np.log1p(y_train_reg_raw) / np.log1p(y_train_reg_raw).max()
)


print("Training Stage 2: regressor...")

reg.fit(
    X_train_reg,
    y_train_reg_log,
    sample_weight=reg_sample_weight,
    eval_set=[(X_val_reg, y_val_reg_log)],
    eval_metric="mae",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)
















train_reg_pred_log = reg.predict(X_train_reg)
train_residuals = y_train_reg_log - train_reg_pred_log

sigma = np.std(train_residuals)
bias_correction = np.exp(0.5 * sigma**2)

print("sigma:", sigma)
print("bias_correction:", bias_correction)

train_income_if_active_log = reg.predict(X_train)
val_income_if_active_log = reg.predict(X_val)

train_income_if_active = np.expm1(train_income_if_active_log) * bias_correction
val_income_if_active = np.expm1(val_income_if_active_log) * bias_correction

train_income_if_active = np.clip(train_income_if_active, 0, None)
val_income_if_active = np.clip(val_income_if_active, 0, None)

train_expected_raw = (train_p_active ** GAMMA) * train_income_if_active
val_expected_raw = (val_p_active ** GAMMA) * val_income_if_active

train_expected_raw[train_p_active < ZERO_THRESHOLD] = 0
val_expected_raw[val_p_active < ZERO_THRESHOLD] = 0

train_expected_raw = np.clip(train_expected_raw, 0, None)
val_expected_raw = np.clip(val_expected_raw, 0, None)











def build_calibration_table(
    X_val,
    y_val_true,
    y_val_pred_raw,
    segment_col=SEGMENT_COL,
    n_deciles=10,
    min_group_size=50,
    factor_min=0.25,
    factor_max=3.0
):
    tmp = pd.DataFrame(index=X_val.index)

    if segment_col in X_val.columns:
        tmp[segment_col] = X_val[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["true"] = y_val_true.values
    tmp["pred_raw"] = np.clip(y_val_pred_raw, 0, None)

    pred_decile, pred_decile_edges = make_qcut_bins(
        tmp["pred_raw"],
        n_bins=n_deciles
    )

    tmp["pred_decile"] = pred_decile

    global_factor = safe_ratio(
        tmp["true"].sum(),
        tmp["pred_raw"].sum(),
        default=1.0
    )

    segment_table = (
        tmp.groupby(segment_col)
        .agg(
            n=("true", "size"),
            true_sum=("true", "sum"),
            pred_sum=("pred_raw", "sum")
        )
        .reset_index()
    )

    segment_table["segment_factor"] = (
        segment_table["true_sum"] /
        segment_table["pred_sum"].replace(0, np.nan)
    )

    segment_table["segment_factor"] = (
        segment_table["segment_factor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_factor)
        .clip(factor_min, factor_max)
    )

    calibration_table = (
        tmp.groupby([segment_col, "pred_decile"])
        .agg(
            n=("true", "size"),
            true_sum=("true", "sum"),
            pred_sum=("pred_raw", "sum"),
            true_mean=("true", "mean"),
            pred_mean=("pred_raw", "mean"),
            true_median=("true", "median"),
            pred_median=("pred_raw", "median")
        )
        .reset_index()
    )

    calibration_table["factor"] = (
        calibration_table["true_sum"] /
        calibration_table["pred_sum"].replace(0, np.nan)
    )

    calibration_table = calibration_table.merge(
        segment_table[[segment_col, "segment_factor"]],
        on=segment_col,
        how="left"
    )

    calibration_table["factor"] = np.where(
        calibration_table["n"] >= min_group_size,
        calibration_table["factor"],
        calibration_table["segment_factor"]
    )

    calibration_table["factor"] = (
        calibration_table["factor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_factor)
        .clip(factor_min, factor_max)
    )

    return calibration_table, segment_table, global_factor, pred_decile_edges

def apply_calibration(
    X_part,
    pred_raw,
    calibration_table,
    segment_table,
    global_factor,
    pred_decile_edges,
    segment_col=SEGMENT_COL
):
    tmp = pd.DataFrame(index=X_part.index)

    if segment_col in X_part.columns:
        tmp[segment_col] = X_part[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["pred_raw"] = np.clip(pred_raw, 0, None)
    tmp["pred_decile"] = apply_fixed_bins(tmp["pred_raw"], pred_decile_edges)

    tmp = tmp.merge(
        calibration_table[[segment_col, "pred_decile", "factor"]],
        on=[segment_col, "pred_decile"],
        how="left"
    )

    tmp = tmp.merge(
        segment_table[[segment_col, "segment_factor"]],
        on=segment_col,
        how="left"
    )

    tmp["factor"] = tmp["factor"].fillna(tmp["segment_factor"])
    tmp["factor"] = tmp["factor"].fillna(global_factor)
    tmp["factor"] = tmp["factor"].fillna(1.0)

    pred_calibrated = pred_raw * tmp["factor"].values
    pred_calibrated = np.clip(pred_calibrated, 0, None)

    return pred_calibrated, tmp["factor"].values, tmp["pred_decile"].values



calibration_table, segment_calibration_table, global_calibration_factor, pred_decile_edges = (
    build_calibration_table(
        X_val=X_val,
        y_val_true=y_val_raw,
        y_val_pred_raw=val_expected_raw,
        segment_col=SEGMENT_COL,
        n_deciles=N_DECILES,
        min_group_size=MIN_GROUP_SIZE_FOR_CALIBRATION,
        factor_min=CALIBRATION_FACTOR_MIN,
        factor_max=CALIBRATION_FACTOR_MAX
    )
)

display(calibration_table)
display(segment_calibration_table)

print("global_calibration_factor:", global_calibration_factor)

calibration_table, segment_calibration_table, global_calibration_factor, pred_decile_edges = (
    build_calibration_table(
        X_val=X_val,
        y_val_true=y_val_raw,
        y_val_pred_raw=val_expected_raw,
        segment_col=SEGMENT_COL,
        n_deciles=N_DECILES,
        min_group_size=MIN_GROUP_SIZE_FOR_CALIBRATION,
        factor_min=CALIBRATION_FACTOR_MIN,
        factor_max=CALIBRATION_FACTOR_MAX
    )
)

display(calibration_table)
display(segment_calibration_table)

print("global_calibration_factor:", global_calibration_factor)













train_expected_calibrated, train_calibration_factor, train_pred_decile = apply_calibration(
    X_part=X_train,
    pred_raw=train_expected_raw,
    calibration_table=calibration_table,
    segment_table=segment_calibration_table,
    global_factor=global_calibration_factor,
    pred_decile_edges=pred_decile_edges,
    segment_col=SEGMENT_COL
)

val_expected_calibrated, val_calibration_factor, val_pred_decile = apply_calibration(
    X_part=X_val,
    pred_raw=val_expected_raw,
    calibration_table=calibration_table,
    segment_table=segment_calibration_table,
    global_factor=global_calibration_factor,
    pred_decile_edges=pred_decile_edges,
    segment_col=SEGMENT_COL
)

def apply_caps(
    X_part,
    pred,
    caps_by_segment,
    global_cap,
    segment_col=SEGMENT_COL
):
    pred = np.array(pred, dtype=float)

    caps = np.full(len(pred), global_cap, dtype=float)

    if segment_col in X_part.columns:
        segments = X_part[segment_col].astype(str).values

        for i, seg in enumerate(segments):
            caps[i] = caps_by_segment.get(seg, global_cap)

    pred_capped = np.minimum(pred, caps)
    pred_capped = np.clip(pred_capped, 0, None)

    return pred_capped, caps

caps_by_segment, global_cap = build_caps_by_segment(
    X_train=X_train,
    y_train=y_train_raw,
    segment_col=SEGMENT_COL,
    active_threshold=ACTIVE_THRESHOLD,
    cap_quantile=CAP_QUANTILE_BY_SEGMENT
)

print("global_cap:", global_cap)
print("caps_by_segment:", caps_by_segment)

train_pred_capped, train_caps_used = apply_caps(
    X_part=X_train,
    pred=train_expected_calibrated,
    caps_by_segment=caps_by_segment,
    global_cap=global_cap,
    segment_col=SEGMENT_COL
)

val_pred_capped, val_caps_used = apply_caps(
    X_part=X_val,
    pred=val_expected_calibrated,
    caps_by_segment=caps_by_segment,
    global_cap=global_cap,
    segment_col=SEGMENT_COL
)

validation_results = pd.DataFrame({
    ID_COL: X_val.index,
    "True_Value": y_val_raw.values,
    "P_ACTIVE": val_p_active,
    "IS_LIKELY_ACTIVE": (val_p_active >= CLASSIFICATION_THRESHOLD).astype(int),
    "Income_If_Active": val_income_if_active,
    "Expected_Raw": val_expected_raw,
    "Calibration_Factor": val_calibration_factor,
    "Pred_Decile": val_pred_decile,
    "Expected_Calibrated": val_expected_calibrated,
    "Cap_Used": val_caps_used,
    "Predicted_Before_Segment_Bucket_Correction": val_pred_capped
}, index=X_val.index)

if SEGMENT_COL in X_val.columns:
    validation_results[SEGMENT_COL] = X_val[SEGMENT_COL].astype(str).values
else:
    validation_results[SEGMENT_COL] = "ALL"

validation_results["Predicted"] = validation_results["Predicted_Before_Segment_Bucket_Correction"]
validation_results["Error"] = validation_results["Predicted"] - validation_results["True_Value"]
validation_results["Abs_Error"] = validation_results["Error"].abs()
validation_results["Ratio"] = (
    validation_results["Predicted"] /
    validation_results["True_Value"].replace(0, np.nan)
)

validation_results.head()






def build_segment_bucket_correction(
    validation_results,
    segment_col=SEGMENT_COL,
    true_col="True_Value",
    pred_col="Predicted",
    n_bins=10,
    min_group_size=50,
    factor_min=0.05,
    factor_max=2.0
):
    df_corr = validation_results.copy()

    df_corr[pred_col] = df_corr[pred_col].clip(lower=0)

    pred_bucket, pred_bucket_edges = make_qcut_bins(
        df_corr[pred_col],
        n_bins=n_bins
    )

    df_corr["pred_bucket"] = pred_bucket

    global_factor = safe_ratio(
        df_corr[true_col].sum(),
        df_corr[pred_col].sum(),
        default=1.0
    )

    segment_fallback = (
        df_corr
        .groupby(segment_col)
        .agg(
            segment_n=(true_col, "size"),
            segment_true_sum=(true_col, "sum"),
            segment_pred_sum=(pred_col, "sum")
        )
        .reset_index()
    )

    segment_fallback["segment_factor"] = (
        segment_fallback["segment_true_sum"] /
        segment_fallback["segment_pred_sum"].replace(0, np.nan)
    )

    segment_fallback["segment_factor"] = (
        segment_fallback["segment_factor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_factor)
        .clip(factor_min, factor_max)
    )

    correction_table = (
        df_corr
        .groupby([segment_col, "pred_bucket"])
        .agg(
            n=(true_col, "size"),
            true_sum=(true_col, "sum"),
            pred_sum=(pred_col, "sum"),
            true_mean=(true_col, "mean"),
            pred_mean=(pred_col, "mean"),
            true_median=(true_col, "median"),
            pred_median=(pred_col, "median")
        )
        .reset_index()
    )

    correction_table["factor"] = (
        correction_table["true_sum"] /
        correction_table["pred_sum"].replace(0, np.nan)
    )

    correction_table = correction_table.merge(
        segment_fallback[[segment_col, "segment_factor"]],
        on=segment_col,
        how="left"
    )

    correction_table["factor"] = np.where(
        correction_table["n"] >= min_group_size,
        correction_table["factor"],
        correction_table["segment_factor"]
    )

    correction_table["factor"] = (
        correction_table["factor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_factor)
        .clip(factor_min, factor_max)
    )

    return correction_table, segment_fallback, global_factor, pred_bucket_edges

def apply_segment_bucket_correction(
    X_part,
    pred,
    correction_table,
    segment_fallback,
    global_factor,
    pred_bucket_edges,
    segment_col=SEGMENT_COL
):
    pred = np.array(pred, dtype=float)

    tmp = pd.DataFrame(index=X_part.index)
    tmp["pred"] = np.clip(pred, 0, None)

    if segment_col in X_part.columns:
        tmp[segment_col] = X_part[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["pred_bucket"] = apply_fixed_bins(tmp["pred"], pred_bucket_edges)

    tmp = tmp.merge(
        correction_table[[segment_col, "pred_bucket", "factor"]],
        on=[segment_col, "pred_bucket"],
        how="left"
    )

    tmp = tmp.merge(
        segment_fallback[[segment_col, "segment_factor"]],
        on=segment_col,
        how="left"
    )

    tmp["factor"] = tmp["factor"].fillna(tmp["segment_factor"])
    tmp["factor"] = tmp["factor"].fillna(global_factor)
    tmp["factor"] = tmp["factor"].fillna(1.0)

    pred_corrected = pred * tmp["factor"].values
    pred_corrected = np.clip(pred_corrected, 0, None)

    return pred_corrected, tmp["factor"].values, tmp["pred_bucket"].values


segment_bucket_correction, segment_bucket_fallback, global_segment_bucket_factor, pred_bucket_edges = (
    build_segment_bucket_correction(
        validation_results=validation_results,
        segment_col=SEGMENT_COL,
        true_col="True_Value",
        pred_col="Predicted",
        n_bins=N_DECILES,
        min_group_size=MIN_GROUP_SIZE_FOR_SEGMENT_BUCKET,
        factor_min=SEGMENT_BUCKET_FACTOR_MIN,
        factor_max=SEGMENT_BUCKET_FACTOR_MAX
    )
)

display(segment_bucket_correction)
display(segment_bucket_fallback)

print("global_segment_bucket_factor:", global_segment_bucket_factor)

val_final_pred, val_segment_bucket_factor, val_pred_bucket = apply_segment_bucket_correction(
    X_part=X_val,
    pred=val_pred_capped,
    correction_table=segment_bucket_correction,
    segment_fallback=segment_bucket_fallback,
    global_factor=global_segment_bucket_factor,
    pred_bucket_edges=pred_bucket_edges,
    segment_col=SEGMENT_COL
)

train_final_pred, train_segment_bucket_factor, train_pred_bucket = apply_segment_bucket_correction(
    X_part=X_train,
    pred=train_pred_capped,
    correction_table=segment_bucket_correction,
    segment_fallback=segment_bucket_fallback,
    global_factor=global_segment_bucket_factor,
    pred_bucket_edges=pred_bucket_edges,
    segment_col=SEGMENT_COL
)





validation_results["Segment_Bucket_Factor"] = val_segment_bucket_factor
validation_results["Segment_Pred_Bucket"] = val_pred_bucket
validation_results["Predicted"] = val_final_pred

validation_results["Error"] = validation_results["Predicted"] - validation_results["True_Value"]
validation_results["Abs_Error"] = validation_results["Error"].abs()
validation_results["Ratio"] = (
    validation_results["Predicted"] /
    validation_results["True_Value"].replace(0, np.nan)
)

round_cols = [
    "P_ACTIVE",
    "Income_If_Active",
    "Expected_Raw",
    "Calibration_Factor",
    "Expected_Calibrated",
    "Cap_Used",
    "Predicted_Before_Segment_Bucket_Correction",
    "Segment_Bucket_Factor",
    "Predicted",
    "Error",
    "Abs_Error",
    "Ratio"
]

for c in round_cols:
    if c in validation_results.columns:
        validation_results[c] = validation_results[c].round(4)

validation_results.head(20)













def regression_metrics(y_true, y_pred, title):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.clip(y_pred, 0, None))

    eps = 1e-9

    print("=" * 80)
    print(title)
    print("-" * 80)
    print(f"MAE       : {mean_absolute_error(y_true, y_pred):,.4f}")
    print(f"MedAE     : {median_absolute_error(y_true, y_pred):,.4f}")
    print(f"R2        : {r2_score(y_true, y_pred):,.4f}")
    print(f"MAE_log   : {mean_absolute_error(y_true_log, y_pred_log):,.4f}")
    print(f"MedAE_log : {median_absolute_error(y_true_log, y_pred_log):,.4f}")
    print(f"R2_log    : {r2_score(y_true_log, y_pred_log):,.4f}")
    print("-" * 80)
    print(f"sum_true  : {y_true.sum():,.4f}")
    print(f"sum_pred  : {y_pred.sum():,.4f}")
    print(f"sum_ratio : {y_pred.sum() / max(y_true.sum(), eps):,.4f}")
    print(f"mean_true : {y_true.mean():,.4f}")
    print(f"mean_pred : {y_pred.mean():,.4f}")
    print(f"bias_mean : {np.mean(y_pred - y_true):,.4f}")
    print(f"bias_sum  : {(y_pred.sum() - y_true.sum()):,.4f}")
    print("=" * 80)


regression_metrics(
    y_val_raw,
    val_pred_capped,
    "[Validation] Before Segment × Prediction Bucket Correction"
)

regression_metrics(
    y_val_raw,
    val_final_pred,
    "[Validation] After Segment × Prediction Bucket Correction"
)

regression_metrics(
    y_train_raw,
    train_final_pred,
    "[Train] Final Prediction"
)










segment_report = (
    validation_results
    .groupby(SEGMENT_COL)
    .agg(
        n=("True_Value", "size"),
        true_sum=("True_Value", "sum"),
        pred_sum_before=("Predicted_Before_Segment_Bucket_Correction", "sum"),
        pred_sum_after=("Predicted", "sum"),
        true_mean=("True_Value", "mean"),
        pred_mean_before=("Predicted_Before_Segment_Bucket_Correction", "mean"),
        pred_mean_after=("Predicted", "mean"),
        true_median=("True_Value", "median"),
        pred_median_before=("Predicted_Before_Segment_Bucket_Correction", "median"),
        pred_median_after=("Predicted", "median"),
        mae_after=("Abs_Error", "mean"),
        p_active_mean=("P_ACTIVE", "mean")
    )
    .reset_index()
)

segment_report["ratio_before"] = (
    segment_report["pred_sum_before"] /
    segment_report["true_sum"].replace(0, np.nan)
)

segment_report["ratio_after"] = (
    segment_report["pred_sum_after"] /
    segment_report["true_sum"].replace(0, np.nan)
)

display(segment_report)




segment_bucket_report = (
    validation_results
    .groupby([SEGMENT_COL, "Segment_Pred_Bucket"])
    .agg(
        n=("True_Value", "size"),
        true_sum=("True_Value", "sum"),
        pred_sum=("Predicted", "sum"),
        true_mean=("True_Value", "mean"),
        pred_mean=("Predicted", "mean"),
        true_median=("True_Value", "median"),
        pred_median=("Predicted", "median"),
        factor=("Segment_Bucket_Factor", "mean")
    )
    .reset_index()
)

segment_bucket_report["sum_ratio"] = (
    segment_bucket_report["pred_sum"] /
    segment_bucket_report["true_sum"].replace(0, np.nan)
)

display(segment_bucket_report)


# MICRO/SMALL, true very low, prediction still high
problem_low_clients = validation_results[
    (validation_results[SEGMENT_COL].isin(["MICRO", "SMALL"])) &
    (validation_results["True_Value"] < 100) &
    (validation_results["Predicted"] > 1000)
].sort_values("Predicted", ascending=False)

problem_low_clients.head(50)
# Biggest overpredictions
validation_results.sort_values("Error", ascending=False).head(50)

# Biggest underpredictions
validation_results.sort_values("Error", ascending=True).head(50)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))

sns.kdeplot(
    np.log1p(validation_results["True_Value"]),
    label="True",
    fill=True,
    alpha=0.25
)

sns.kdeplot(
    np.log1p(validation_results["Predicted"]),
    label="Predicted",
    linestyle="--"
)

plt.xlabel("log1p income")
plt.ylabel("Density")
plt.title("Distribution Match: True vs Final Prediction")
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))

plt.scatter(
    np.log1p(validation_results["True_Value"]),
    np.log1p(validation_results["Predicted"]),
    alpha=0.25,
    s=15
)

mx = max(
    np.log1p(validation_results["True_Value"]).max(),
    np.log1p(validation_results["Predicted"]).max()
)

plt.plot([0, mx], [0, mx], "--")

plt.xlabel("True log1p income")
plt.ylabel("Predicted log1p income")
plt.title("True vs Predicted")
plt.show()








validation_results_view = validation_results[
    [
        ID_COL,
        SEGMENT_COL,
        "True_Value",
        "P_ACTIVE",
        "Income_If_Active",
        "Expected_Raw",
        "Expected_Calibrated",
        "Cap_Used",
        "Predicted_Before_Segment_Bucket_Correction",
        "Segment_Bucket_Factor",
        "Segment_Pred_Bucket",
        "Predicted",
        "Error",
        "Abs_Error",
        "Ratio"
    ]
].copy()

validation_results_view.head(30)











model_artifacts = {
    "clf": clf,
    "reg": reg,

    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "cat_values": cat_values,

    "ACTIVE_THRESHOLD": ACTIVE_THRESHOLD,
    "CLASSIFICATION_THRESHOLD": CLASSIFICATION_THRESHOLD,
    "GAMMA": GAMMA,
    "ZERO_THRESHOLD": ZERO_THRESHOLD,
    "bias_correction": bias_correction,

    "SEGMENT_COL": SEGMENT_COL,
    "N_DECILES": N_DECILES,

    "calibration_table": calibration_table,
    "segment_calibration_table": segment_calibration_table,
    "global_calibration_factor": global_calibration_factor,
    "pred_decile_edges": pred_decile_edges,

    "caps_by_segment": caps_by_segment,
    "global_cap": global_cap,

    "segment_bucket_correction": segment_bucket_correction,
    "segment_bucket_fallback": segment_bucket_fallback,
    "global_segment_bucket_factor": global_segment_bucket_factor,
    "pred_bucket_edges": pred_bucket_edges
}