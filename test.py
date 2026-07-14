def shift_month(month, n):
    period = pd.Period(month.replace("_", "-"), freq="M")
    return (period + n).strftime("%Y_%m")


def build_validation_month(score_month):
    scores = read_scores(score_month)

    t1_month = shift_month(score_month, 1)
    fx_future_months = [
        shift_month(score_month, i)
        for i in range(1, 6)
    ]

    # Дохід у місяці формування propensity-скорів
    income_t = (
        read_income(score_month)
        .rename(columns={
            "INCOME_LIABILITIES": "INCOME_LIABILITIES_T",
            "INCOME_ASSETS": "INCOME_ASSETS_T",
            "INCOME_FX": "INCOME_FX_T"
        })
    )

    # Для Liabilities та Assets дивимося лише T+1
    income_t1 = (
        read_income(t1_month)
        [["CONTRAGENTID", "INCOME_LIABILITIES", "INCOME_ASSETS"]]
        .rename(columns={
            "INCOME_LIABILITIES": "INCOME_LIABILITIES_T1",
            "INCOME_ASSETS": "INCOME_ASSETS_T1"
        })
    )

    df = (
        scores
        .merge(income_t, how="left", on="CONTRAGENTID")
        .merge(income_t1, how="left", on="CONTRAGENTID")
    )

    # FX: доходи за T+1 ... T+5
    fx_future_cols = []

    for i, month in enumerate(fx_future_months, start=1):
        col = f"INCOME_FX_T{i}"

        fx_month = (
            read_income(month)
            [["CONTRAGENTID", "INCOME_FX"]]
            .rename(columns={"INCOME_FX": col})
        )

        df = df.merge(fx_month, how="left", on="CONTRAGENTID")
        fx_future_cols.append(col)

    income_cols = [
        "INCOME_LIABILITIES_T",
        "INCOME_ASSETS_T",
        "INCOME_FX_T",
        "INCOME_LIABILITIES_T1",
        "INCOME_ASSETS_T1",
        *fx_future_cols
    ]

    df[income_cols] = df[income_cols].fillna(0)

    df["NEW_LIABILITIES"] = (
        (df["INCOME_LIABILITIES_T"] <= 0)
        & (df["INCOME_LIABILITIES_T1"] > 0)
    )

    df["NEW_ASSETS"] = (
        (df["INCOME_ASSETS_T"] <= 0)
        & (df["INCOME_ASSETS_T1"] > 0)
    )

    df["NEW_FX"] = (
        (df["INCOME_FX_T"] <= 0)
        & df[fx_future_cols].gt(0).any(axis=1)
    )

    product_flags = {
        "Liabilities": "NEW_LIABILITIES",
        "Assets": "NEW_ASSETS",
        "FX": "NEW_FX"
    }

    df["actual_product"] = df.apply(
        lambda row: ", ".join(
            product
            for product, flag in product_flags.items()
            if row[flag]
        ) or "None",
        axis=1
    )

    df["n_actual_products"] = df[
        list(product_flags.values())
    ].sum(axis=1)

    df["score_month"] = score_month
    df["income_month"] = t1_month
    df["fx_horizon_end"] = fx_future_months[-1]

    return df









score_months = [
    "2025_06",
    "2025_07",
    "2025_08",
    # додай потрібні місяці
]

validation_parts = []

for score_month in score_months:
    try:
        month_df = build_validation_month(score_month)
        validation_parts.append(month_df)

        print(f"{score_month}: OK, rows = {len(month_df):,}")

    except Exception as e:
        print(f"{score_month}: ERROR -> {e}")

validation_all_full = pd.concat(
    validation_parts,
    ignore_index=True
)




def sample_validation_negatives(
    df,
    negative_share=0.50,
    random_state=42
):
    parts = []

    for _, month_df in df.groupby("score_month"):
        positives = month_df[
            month_df["n_actual_products"] > 0
        ]

        negatives = month_df[
            month_df["n_actual_products"] == 0
        ]

        # Скільки негативних потрібно для заданої частки
        n_negatives = round(
            len(positives)
            * negative_share
            / (1 - negative_share)
        )

        negatives = negatives.sample(
            n=min(n_negatives, len(negatives)),
            random_state=random_state
        )

        parts.append(
            pd.concat([positives, negatives])
        )

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


validation_all = sample_validation_negatives(
    validation_all_full,
    negative_share=0.50
)


validation_all["actual_product"].value_counts(dropna=False)
validation_all["actual_product"].value_counts(normalize=True)