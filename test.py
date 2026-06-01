rec = (
    liabs[["IDENTIFYCODE", "PRIMARY"]]
    .rename(columns={"PRIMARY": "p_liabs"})
    .merge(
        assets[["IDENTIFYCODE", "PRIMARY"]].rename(columns={"PRIMARY": "p_assets"}),
        on="IDENTIFYCODE",
        how="outer"
    )
    .merge(
        fx[["IDENTIFYCODE", "PROB_TO_FX"]].rename(columns={"PROB_TO_FX": "p_fx"}),
        on="IDENTIFYCODE",
        how="outer"
    )
)

rec[["p_liabs", "p_assets", "p_fx"]] = rec[["p_liabs", "p_assets", "p_fx"]].fillna(0)

rec["recommended_product"] = rec[["p_liabs", "p_assets", "p_fx"]].idxmax(axis=1)

rec["recommended_product"] = rec["recommended_product"].map({
    "p_liabs": "Liabilities",
    "p_assets": "Assets",
    "p_fx": "FX"
})

rec["recommendation_score"] = rec[["p_liabs", "p_assets", "p_fx"]].max(axis=1)