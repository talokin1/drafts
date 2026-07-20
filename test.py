import numpy as np
import pandas as pd

df = dataset.copy()

SCORES = {
    "LIABILITIES": ("LIAB_PRIMARY", 0.30),
    "ASSETS": ("ASSETS_PRIMARY", 0.30),
    "FX": ("FX_PRIMARY", 0.59),
}

# Залишаємо клієнтів, для яких є хоча б один propensity score
score_cols = [x[0] for x in SCORES.values()]

df = df[
    df[score_cols].notna().any(axis=1)
].copy()


# 1. Фактичне використання продуктів
df["USE_LIABILITIES"] = df["AV_LIABILITIES"].fillna(0).gt(0)
df["USE_ASSETS"] = df["CONTRACTED_AMOUNT_LCY"].fillna(0).gt(0)
df["USE_FX"] = df["FX_NB_6M"].fillna(0).gt(0)

use_cols = [
    "USE_LIABILITIES",
    "USE_ASSETS",
    "USE_FX",
]

df["USED_ANY_PRODUCT"] = df[use_cols].any(axis=1)


# 2. Multi-label target
def get_actual_product(row):
    products = []

    if row["USE_LIABILITIES"]:
        products.append("LIABILITIES")

    if row["USE_ASSETS"]:
        products.append("ASSETS")

    if row["USE_FX"]:
        products.append("FX")

    return ", ".join(products) if products else "NOTHING_TO_DO"


df["ACTUAL_PRODUCT"] = df.apply(
    get_actual_product,
    axis=1,
)


# 3. Високий propensity score без залучення
df["HIGH_PRIMARY"] = (
    (df["LIAB_PRIMARY"] >= 0.30)
    | (df["ASSETS_PRIMARY"] >= 0.30)
    | (df["FX_PRIMARY"] >= 0.59)
)


# 4. Тип запису
df["RECORD_TYPE"] = np.select(
    [
        df["USED_ANY_PRODUCT"],
        ~df["USED_ANY_PRODUCT"] & df["HIGH_PRIMARY"],
        ~df["USED_ANY_PRODUCT"] & ~df["HIGH_PRIMARY"],
    ],
    [
        "POSITIVE",
        "BAD_RECORD",
        "NOTHING_TO_DO",
    ],
)


df["RECORD_TYPE"].value_counts()



model_dataset = df[
    df["RECORD_TYPE"] != "BAD_RECORD"
][
    [
        "CONTRAGENTID",
        "IDENTIFYCODE",
        "LIAB_PRIMARY",
        "ASSETS_PRIMARY",
        "FX_PRIMARY",
        "ACTUAL_PRODUCT",
        "RECORD_TYPE",
    ]
].copy()