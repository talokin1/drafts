def apply_calibration(
    X_part,
    pred_raw,
    calibration_table,
    segment_calibration_table,
    global_factor,
    segment_col=SEGMENT_COL,
    n_deciles=10
):
    tmp = pd.DataFrame(index=X_part.index)
    
    if segment_col in X_part.columns:
        tmp[segment_col] = X_part[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["pred_raw"] = pred_raw
    tmp["pred_decile"] = make_prediction_deciles(tmp["pred_raw"], n_deciles=n_deciles).values

    tmp = tmp.merge(
        calibration_table[[segment_col, "pred_decile", "factor"]],
        on=[segment_col, "pred_decile"],
        how="left"
    )

    tmp = tmp.merge(
        segment_calibration_table[[segment_col, "factor"]].rename(columns={"factor": "segment_factor"}),
        on=segment_col,
        how="left"
    )

    tmp["factor"] = tmp["factor"].fillna(tmp["segment_factor"])
    tmp["factor"] = tmp["factor"].fillna(global_factor)
    tmp["factor"] = tmp["factor"].fillna(1.0)

    calibrated = pred_raw * tmp["factor"].values

    return calibrated, tmp["factor"].values