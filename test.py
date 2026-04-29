def build_caps_by_segment(
    X_train,
    y_train,
    segment_col=SEGMENT_COL,
    active_threshold=ACTIVE_THRESHOLD,
    cap_quantile=0.99
):
    tmp = X_train[[segment_col]].copy() if segment_col in X_train.columns else pd.DataFrame(index=X_train.index)

    if segment_col not in tmp.columns:
        tmp[segment_col] = "ALL"

    tmp["target"] = y_train.values

    tmp_active = tmp[tmp["target"] > active_threshold].copy()

    caps = (
        tmp_active.groupby(segment_col)["target"]
        .quantile(cap_quantile)
        .to_dict()
    )

    global_cap = tmp_active["target"].quantile(cap_quantile)

    if pd.isna(global_cap):
        global_cap = y_train.quantile(cap_quantile)

    return caps, global_cap