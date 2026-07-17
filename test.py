score_parts = []

for month in SCORE_MONTHS:
    try:
        part = read_scores(month).copy()
        part["score_month"] = month
        score_parts.append(part)

        print(f"{month}: scores OK, rows = {len(part):,}")

    except FileNotFoundError:
        print(f"{month}: score files not found")

scores_all = pd.concat(score_parts, ignore_index=True)

def get_latest_score(df, score_col):
    return (
        df.loc[
            df[score_col].notna(),
            ["IDENTIFYCODE", score_col, "score_month"]
        ]
        .sort_values("score_month")
        .drop_duplicates("IDENTIFYCODE", keep="last")
        .drop(columns="score_month")
    )


scores = client_map.copy()

for score_col in ["LIAB_PRIMARY", "ASSETS_PRIMARY", "FX_PRIMARY"]:
    scores = scores.merge(
        get_latest_score(scores_all, score_col),
        how="left",
        on="IDENTIFYCODE"
    )

scores = scores.dropna(
    subset=["LIAB_PRIMARY", "ASSETS_PRIMARY", "FX_PRIMARY"],
    how="all"
)


income_parts = []

for month in INCOME_MONTHS:
    try:
        part = read_income(month).copy()
        income_parts.append(part)

        print(f"{month}: income OK, rows = {len(part):,}")

    except FileNotFoundError:
        print(f"{month}: income file not found")

income_all = pd.concat(income_parts, ignore_index=True)


income_usage = (
    income_all
    .groupby("CONTRAGENTID", as_index=False)
    .agg(
        USED_LIABILITIES=(
            "INCOME_LIABILITIES",
            lambda x: x.fillna(0).gt(0).any()
        ),
        USED_ASSETS=(
            "INCOME_ASSETS",
            lambda x: x.fillna(0).gt(0).any()
        )
    )
)


dataset_exp = pd.read_csv(
    r"M:\Controlling\Data_Science_Projects\Corp_Liabilities\Data\dataset_2026_06_wo_income.csv",
    dtype={
        "CONTRAGENTID": "string",
        "IDENTIFYCODE": "string"
    }
)

dataset_exp = normalize_identifycode(dataset_exp)
dataset_exp = normalize_contragentid(dataset_exp)

dataset_exp["FX_NB_6M"] = pd.to_numeric(
    dataset_exp["FX_NB_6M"],
    errors="coerce"
).fillna(0)

fx_usage = (
    dataset_exp
    .groupby(
        ["IDENTIFYCODE", "CONTRAGENTID"],
        as_index=False
    )
    .agg(
        USED_FX=("FX_NB_6M", lambda x: x.gt(0).any())
    )
)


validation_all = (
    scores
    .merge(
        income_usage,
        how="left",
        on="CONTRAGENTID"
    )
    .merge(
        fx_usage,
        how="left",
        on=["IDENTIFYCODE", "CONTRAGENTID"]
    )
)

target_cols = [
    "USED_LIABILITIES",
    "USED_ASSETS",
    "USED_FX"
]

validation_all[target_cols] = (
    validation_all[target_cols]
    .fillna(False)
    .astype(bool)
)


PRODUCT_FLAGS = {
    "Liabilities": "USED_LIABILITIES",
    "Assets": "USED_ASSETS",
    "FX": "USED_FX"
}


def get_actual_products(row):
    products = [
        product
        for product, flag_col in PRODUCT_FLAGS.items()
        if row[flag_col]
    ]

    return ", ".join(products) if products else "None"


validation_all["actual_product"] = validation_all.apply(
    get_actual_products,
    axis=1
)

validation_all["n_actual_products"] = (
    validation_all[target_cols]
    .sum(axis=1)
)


validation_all = validation_all[
    [
        "IDENTIFYCODE",
        "CONTRAGENTID",

        "LIAB_PRIMARY",
        "ASSETS_PRIMARY",
        "FX_PRIMARY",

        "USED_LIABILITIES",
        "USED_ASSETS",
        "USED_FX",

        "actual_product",
        "n_actual_products"
    ]
]