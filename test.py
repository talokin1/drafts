import pandas as pd
import numpy as np

# Нормалізація ідентифікаторів
def normalize_ids(df):
    df = df.copy()

    df["CONTRAGENTID"] = (
        df["CONTRAGENTID"].astype("string").str.replace(r"\.0$", "", regex=True)
    )

    df["IDENTIFYCODE"] = (
        df["IDENTIFYCODE"].astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )

    return df


income = normalize_ids(income)
fx_clients = normalize_ids(fx_clients)

liabilities_scores = normalize_ids(liabilities_scores)
assets_scores = normalize_ids(assets_scores)
fx_scores = normalize_ids(fx_scores)





scores = (
    liabilities_scores[
        ["CONTRAGENTID", "IDENTIFYCODE", "PRIMARY"]
    ].rename(columns={"PRIMARY": "LIABILITIES_PRIMARY"})

    .merge(
        assets_scores[
            ["CONTRAGENTID", "IDENTIFYCODE", "PRIMARY"]
        ].rename(columns={"PRIMARY": "ASSETS_PRIMARY"}),
        on=["CONTRAGENTID", "IDENTIFYCODE"],
        how="outer"
    )

    .merge(
        fx_scores[
            ["CONTRAGENTID", "IDENTIFYCODE", "PRIMARY"]
        ].rename(columns={"PRIMARY": "FX_PRIMARY"}),
        on=["CONTRAGENTID", "IDENTIFYCODE"],
        how="outer"
    )
)





usage = (
    income.groupby(["CONTRAGENTID", "IDENTIFYCODE"], as_index=False)
    .agg(
        USED_LIABILITIES=("LIABILITIES_INCOME", lambda x: (x.fillna(0) > 0).any()),
        USED_ASSETS=("ASSETS_INCOME", lambda x: (x.fillna(0) > 0).any())
    )
)

fx_usage = (
    fx_clients.groupby(["CONTRAGENTID", "IDENTIFYCODE"], as_index=False)
    ["FX_NB_6M"].max()
)

fx_usage["USED_FX"] = fx_usage["FX_NB_6M"].fillna(0) > 0
fx_usage = fx_usage.drop(columns="FX_NB_6M")

usage = usage.merge(
    fx_usage,
    on=["CONTRAGENTID", "IDENTIFYCODE"],
    how="outer"
)

usage[["USED_LIABILITIES", "USED_ASSETS", "USED_FX"]] = (
    usage[["USED_LIABILITIES", "USED_ASSETS", "USED_FX"]]
    .fillna(False)
)


validation_dataset = scores.merge(
    usage,
    on=["CONTRAGENTID", "IDENTIFYCODE"],
    how="inner"
)

product_flags = {
    "Liabilities": "USED_LIABILITIES",
    "Assets": "USED_ASSETS",
    "FX": "USED_FX"
}

validation_dataset["USED_PRODUCT"] = validation_dataset.apply(
    lambda row: ", ".join(
        product for product, flag in product_flags.items()
        if row[flag]
    ) or np.nan,
    axis=1
)

validation_dataset = validation_dataset[
    [
        "IDENTIFYCODE",
        "CONTRAGENTID",
        "LIABILITIES_PRIMARY",
        "ASSETS_PRIMARY",
        "FX_PRIMARY",
        "USED_PRODUCT"
    ]
]


validation_dataset["USED_PRODUCT"].value_counts(dropna=False)

final_columns = [
    "IDENTIFYCODE",
    "CONTRAGENTID",
    "LIABILITIES_PRIMARY",
    "ASSETS_PRIMARY",
    "FX_PRIMARY",
    "USED_LIABILITIES",
    "USED_ASSETS",
    "USED_FX",
    "USED_PRODUCT"
]

validation_dataset = validation_dataset[final_columns]