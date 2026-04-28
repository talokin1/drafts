def build_segment_bucket_correction(
    validation_results,
    segment_col="FIRM_TYPE",
    true_col="True_Value",
    pred_col="Predicted",
    n_bins=10,
    min_group_size=50,
    factor_min=0.05,
    factor_max=2.00
):
    df_corr = validation_results.copy()

    df_corr[pred_col] = df_corr[pred_col].clip(lower=0)

    # bin edges фіксуємо на validation
    _, pred_bin_edges = pd.qcut(
        df_corr[pred_col],
        q=n_bins,
        retbins=True,
        duplicates="drop"
    )

    # прибираємо дублікати на всякий випадок
    pred_bin_edges = np.unique(pred_bin_edges)

    # якщо через багато однакових значень bins зламались
    if len(pred_bin_edges) < 2:
        pred_bin_edges = np.array([0, df_corr[pred_col].max() + 1])

    df_corr["pred_bucket"] = pd.cut(
        df_corr[pred_col],
        bins=pred_bin_edges,
        labels=False,
        include_lowest=True
    )

    df_corr["pred_bucket"] = df_corr["pred_bucket"].fillna(0).astype(int)

    # correction по FIRM_TYPE × pred_bucket
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

    # fallback correction тільки по сегменту
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

    global_factor = (
        df_corr[true_col].sum() /
        max(df_corr[pred_col].sum(), 1e-9)
    )

    correction_table = correction_table.merge(
        segment_fallback[[segment_col, "segment_factor"]],
        on=segment_col,
        how="left"
    )

    # якщо група мала — використовуємо segment fallback
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

    segment_fallback["segment_factor"] = (
        segment_fallback["segment_factor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_factor)
        .clip(factor_min, factor_max)
    )

    return correction_table, segment_fallback, global_factor, pred_bin_edges



segment_bucket_correction, segment_correction_fallback, global_segment_bucket_factor, pred_bin_edges = (
    build_segment_bucket_correction(
        validation_results=validation_results,
        segment_col=SEGMENT_COL,
        true_col="True_Value",
        pred_col="Predicted",
        n_bins=10,
        min_group_size=50,
        factor_min=0.05,
        factor_max=2.00
    )
)

display(segment_bucket_correction)
display(segment_correction_fallback)

print("Global correction factor:", global_segment_bucket_factor)
print("Pred bin edges:", pred_bin_edges)


def apply_segment_bucket_correction(
    df_part,
    pred,
    correction_table,
    segment_fallback,
    global_factor,
    pred_bin_edges,
    segment_col="FIRM_TYPE"
):
    pred = np.array(pred, dtype=float)

    tmp = pd.DataFrame(index=df_part.index)
    tmp["pred_before_segment_bucket_correction"] = pred

    if segment_col in df_part.columns:
        tmp[segment_col] = df_part[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["pred_bucket"] = pd.cut(
        tmp["pred_before_segment_bucket_correction"],
        bins=pred_bin_edges,
        labels=False,
        include_lowest=True
    )

    tmp["pred_bucket"] = tmp["pred_bucket"].fillna(0).astype(int)

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

val_final_pred_corrected, val_segment_bucket_factor, val_pred_bucket = (
    apply_segment_bucket_correction(
        df_part=X_val,
        pred=val_final_pred,
        correction_table=segment_bucket_correction,
        segment_fallback=segment_correction_fallback,
        global_factor=global_segment_bucket_factor,
        pred_bin_edges=pred_bin_edges,
        segment_col=SEGMENT_COL
    )
)

validation_results["Predicted_Before_Segment_Bucket_Correction"] = validation_results["Predicted"]
validation_results["Segment_Bucket_Factor"] = val_segment_bucket_factor
validation_results["Pred_Bucket"] = val_pred_bucket
validation_results["Predicted"] = val_final_pred_corrected.round(2)

validation_results["Error"] = validation_results["Predicted"] - validation_results["True_Value"]
validation_results["Abs_Error"] = validation_results["Error"].abs()
validation_results["Ratio"] = (
    validation_results["Predicted"] /
    validation_results["True_Value"].replace(0, np.nan)
)




model_artifacts = {
    "clf": clf,
    "reg": reg,

    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "cat_values": cat_values,

    "calibration_table": calibration_table,
    "segment_calibration_table": segment_calibration_table,
    "global_calibration_factor": global_calibration_factor,

    "caps_by_segment": caps_by_segment,
    "global_cap": global_cap,

    "SEGMENT_COL": SEGMENT_COL,
    "N_DECILES": N_DECILES,

    "GAMMA": GAMMA,
    "ZERO_THRESHOLD": ZERO_THRESHOLD,
    "bias_correction": bias_correction,

    # новий correction layer
    "segment_bucket_correction": segment_bucket_correction,
    "segment_correction_fallback": segment_correction_fallback,
    "global_segment_bucket_factor": global_segment_bucket_factor,
    "pred_bin_edges": pred_bin_edges
}

joblib.dump(model_artifacts, "expected_income_v1_segment_bucket_corrected.pkl")

print("Saved")















pred, segment_bucket_factor, pred_bucket = apply_segment_bucket_correction(
    df_part=df,
    pred=pred,
    correction_table=model["segment_bucket_correction"],
    segment_fallback=model["segment_correction_fallback"],
    global_factor=model["global_segment_bucket_factor"],
    pred_bin_edges=model["pred_bin_edges"],
    segment_col=model["SEGMENT_COL"]
)


df["LIABILITIES_POTENTIAL"] = pred
df["SEGMENT_BUCKET_FACTOR"] = segment_bucket_factor
df["PRED_BUCKET"] = pred_bucket


# ========= CAPS =========

caps = np.full(len(pred), global_cap)

if SEGMENT_COL in df.columns:
    segments = df[SEGMENT_COL].astype(str).values

    for i, seg in enumerate(segments):
        caps[i] = caps_by_segment.get(seg, global_cap)

pred = np.minimum(pred, caps)
pred = np.clip(pred, 0, None)


# ========= SEGMENT × PREDICTION BUCKET CORRECTION =========

pred, segment_bucket_factor, pred_bucket = apply_segment_bucket_correction(
    df_part=df,
    pred=pred,
    correction_table=model["segment_bucket_correction"],
    segment_fallback=model["segment_correction_fallback"],
    global_factor=model["global_segment_bucket_factor"],
    pred_bin_edges=model["pred_bin_edges"],
    segment_col=SEGMENT_COL
)


# ========= FINAL =========

df["LIABILITIES_POTENTIAL"] = pred
df["SEGMENT_BUCKET_FACTOR"] = segment_bucket_factor
df["PRED_BUCKET"] = pred_bucket