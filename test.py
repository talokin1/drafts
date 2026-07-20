invalid_mask = (
    (model_dataset["ACTUAL_PRODUCT"].str.contains("LIABILITIES", na=False) &
     model_dataset["LIAB_PRIMARY"].isna())
    |
    (model_dataset["ACTUAL_PRODUCT"].str.contains("ASSETS", na=False) &
     model_dataset["ASSETS_PRIMARY"].isna())
    |
    (model_dataset["ACTUAL_PRODUCT"].str.contains("FX", na=False) &
     model_dataset["FX_PRIMARY"].isna())
)

model_dataset = model_dataset.loc[~invalid_mask].copy()