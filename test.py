keys = [
    "score_month",
    "IDENTIFYCODE",
    "CONTRAGENTID"
]

# Propensity-скори у широкий формат
scores_wide = (
    validation_all_full
    .pivot_table(
        index=keys,
        columns="scored_product",
        values="propensity_score",
        aggfunc="max"
    )
    .reindex(
        columns=["Liabilities", "Assets", "FX"]
    )
    .reset_index()
    .rename(columns={
        "Liabilities": "LIAB_PRIMARY",
        "Assets": "ASSETS_PRIMARY",
        "FX": "FX_PRIMARY"
    })
)

# Фактичні залучення
targets = (
    validation_all_full
    .groupby(keys, as_index=False)
    [[
        "NEW_LIABILITIES",
        "NEW_ASSETS",
        "NEW_FX"
    ]]
    .max()
)

validation_model = scores_wide.merge(
    targets,
    how="left",
    on=keys
)

TARGET_MAPPING = {
    "Liabilities": "NEW_LIABILITIES",
    "Assets": "NEW_ASSETS",
    "FX": "NEW_FX"
}

validation_model["actual_product"] = (
    validation_model.apply(
        lambda row: ", ".join(
            product
            for product, target_col
            in TARGET_MAPPING.items()
            if row[target_col]
        ) or "None",
        axis=1
    )
)





PROPENSITY_RULES = {
    "Liabilities": ("LIAB_PRIMARY", 0.30),
    "Assets": ("ASSETS_PRIMARY", 0.30),
    "FX": ("FX_PRIMARY", 0.59)
}

for product, (score_col, threshold) in PROPENSITY_RULES.items():
    validation_model[f"REC_{product.upper()}"] = (
        validation_model[score_col]
        .ge(threshold)
        .fillna(False)
    )


validation_model["recommended_product"] = (
    validation_model.apply(
        lambda row: ", ".join(
            product
            for product in PROPENSITY_RULES
            if row[f"REC_{product.upper()}"]
        ) or "None",
        axis=1
    )
)














validation_model[
    "recommended_product"
].value_counts(normalize=True)