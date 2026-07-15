import numpy as np
import pandas as pd


# ============================================================
# Нормалізація ідентифікаторів
# ============================================================

def normalize_identifycode(df, col="IDENTIFYCODE"):
    df = df.copy()

    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )

    return df


def normalize_contragentid(df, col="CONTRAGENTID"):
    df = df.copy()

    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    return df


def to_period(month):
    return pd.Period(
        month.replace("_", "-"),
        freq="M"
    )





INCOME_MONTHS = [
    "2025_06", "2025_07", "2025_08",
    "2025_09", "2025_10", "2025_11",
    "2025_12", "2026_01", "2026_02",
    "2026_03", "2026_04", "2026_05",
    "2026_06"
]


SCORE_COLUMNS = {
    "LIAB_PRIMARY": "Liabilities",
    "ASSETS_PRIMARY": "Assets",
    "FX_PRIMARY": "FX"
}


INCOME_COLUMNS = {
    "Liabilities": "INCOME_LIABILITIES",
    "Assets": "INCOME_ASSETS",
    "FX": "INCOME_FX"
}


NEW_COLUMNS = {
    "Liabilities": "NEW_LIABILITIES",
    "Assets": "NEW_ASSETS",
    "FX": "NEW_FX"
}





income_cache = {}

for month in INCOME_MONTHS:
    income_month = read_income(month)
    income_month = normalize_contragentid(income_month)

    income_cache[month] = income_month


def build_validation_month(score_month):
    scores = read_scores(score_month)

    scores = normalize_identifycode(scores)
    scores = normalize_contragentid(scores)

    # Один рядок = клієнт + продукт скорингу
    score_long = (
        scores
        .melt(
            id_vars=[
                "IDENTIFYCODE",
                "CONTRAGENTID"
            ],
            value_vars=list(SCORE_COLUMNS),
            var_name="score_column",
            value_name="propensity_score"
        )
        .dropna(subset=["propensity_score"])
    )

    score_long["scored_product"] = (
        score_long["score_column"]
        .map(SCORE_COLUMNS)
    )

    score_long["score_month"] = score_month

    future_months = [
        month
        for month in INCOME_MONTHS
        if to_period(month) > to_period(score_month)
    ]

    # Для останнього income-місяця майбутнє ще невідоме
    if not future_months:
        return None

    df = score_long.copy()

    # Перевіряємо залучення до кожного продукту
    for product, income_col in INCOME_COLUMNS.items():
        current_income_col = f"{product.upper()}_INCOME_T"
        first_month_col = f"{product.upper()}_FIRST_INCOME_MONTH"
        new_col = NEW_COLUMNS[product]

        # Дохід у місяці скорингу n
        current_income = (
            income_cache[score_month]
            [["CONTRAGENTID", income_col]]
            .rename(columns={
                income_col: current_income_col
            })
        )

        df = df.merge(
            current_income,
            how="left",
            on="CONTRAGENTID"
        )

        df[current_income_col] = (
            df[current_income_col].fillna(0)
        )

        # Перший майбутній місяць із доходом
        future_positive_parts = []

        for future_month in future_months:
            income_future = income_cache[future_month]

            positive = income_future.loc[
                income_future[income_col] > 0,
                ["CONTRAGENTID"]
            ].drop_duplicates()

            positive[first_month_col] = future_month
            future_positive_parts.append(positive)

        if future_positive_parts:
            first_income = (
                pd.concat(
                    future_positive_parts,
                    ignore_index=True
                )
                .sort_values(first_month_col)
                .drop_duplicates(
                    subset="CONTRAGENTID",
                    keep="first"
                )
            )

            df = df.merge(
                first_income,
                how="left",
                on="CONTRAGENTID"
            )

        else:
            df[first_month_col] = pd.NA

        # У місяці n доходу не було,
        # але в одному з майбутніх місяців він з'явився
        df[new_col] = (
            (df[current_income_col] <= 0)
            & df[first_month_col].notna()
        )

    # Чи залучився саме той продукт,
    # у скоринговому звіті якого був клієнт
    df["successful_engagement"] = False
    df["engagement_month"] = pd.NA

    for product in INCOME_COLUMNS:
        product_mask = (
            df["scored_product"] == product
        )

        new_col = NEW_COLUMNS[product]
        first_month_col = (
            f"{product.upper()}_FIRST_INCOME_MONTH"
        )

        df.loc[
            product_mask,
            "successful_engagement"
        ] = df.loc[
            product_mask,
            new_col
        ]

        success_mask = (
            product_mask
            & df[new_col]
        )

        df.loc[
            success_mask,
            "engagement_month"
        ] = df.loc[
            success_mask,
            first_month_col
        ]

    # Усі продукти, які клієнт активував.
    # Продукт поточного скорингового звіту ставимо першим.
    def get_actual_products(row):
        products = [
            product
            for product, new_col
            in NEW_COLUMNS.items()
            if row[new_col]
        ]

        scored_product = row["scored_product"]

        if scored_product in products:
            products.remove(scored_product)
            products.insert(0, scored_product)

        return ", ".join(products) or "None"

    df["actual_product"] = df.apply(
        get_actual_products,
        axis=1
    )

    df["n_actual_products"] = df[
        list(NEW_COLUMNS.values())
    ].sum(axis=1)

    df["followup_months"] = len(future_months)
    df["horizon_end"] = future_months[-1]

    return df






















SCORE_MONTHS = [
    "2025_06", "2025_07", "2025_08",
    "2025_09", "2025_10", "2025_11",
    "2025_12", "2026_01", "2026_02",
    "2026_03", "2026_04", "2026_05",
    "2026_06"
]


validation_parts = []

for score_month in SCORE_MONTHS:
    try:
        month_df = build_validation_month(
            score_month
        )

        if month_df is None:
            print(
                f"{score_month}: SKIPPED — "
                "немає майбутніх income-місяців"
            )
            continue

        validation_parts.append(month_df)

        print(
            f"{score_month}: OK, "
            f"rows = {len(month_df):,}"
        )

    except Exception as e:
        print(f"{score_month}: ERROR -> {e}")


validation_all_full = pd.concat(
    validation_parts,
    ignore_index=True
)

validation_all_full[
    [
        "score_month",
        "IDENTIFYCODE",
        "scored_product",
        "propensity_score",
        "successful_engagement",
        "engagement_month",
        "actual_product",
        "followup_months"
    ]
].head()


validation_all_full.groupby(
    "scored_product"
)["successful_engagement"].value_counts()