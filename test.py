validation_result = validation_dataset.copy()

validation_result = validation_result.rename(
    columns={
        "LIAB_PRIMARY": "p_liabs",
        "ASSETS_PRIMARY": "p_assets",
        "FX_PRIMARY": "p_fx"
    }
)

validation_result["score_liabs"] = validation_result["p_liabs"].apply(
    lambda x: scale_by_threshold(x, thresholds["p_liabs"])
)

validation_result["score_assets"] = validation_result["p_assets"].apply(
    lambda x: scale_by_threshold(x, thresholds["p_assets"])
)

validation_result["score_fx"] = validation_result["p_fx"].apply(
    lambda x: scale_by_threshold(x, thresholds["p_fx"])
)