LAST_INCOME_MONTH = "2026_06"


def month_period(month):
    return pd.Period(
        month.replace("_", "-"),
        freq="M"
    )


def shift_month(month, n):
    return (
        month_period(month) + n
    ).strftime("%Y_%m")


def income_month_available(month):
    return (
        month_period(month)
        <= month_period(LAST_INCOME_MONTH)
    )


def build_validation_month(score_month):
    scores = read_scores(score_month)

    # Поточний дохід T
    income_t = (
        read_income(score_month)
        .rename(columns={
            "INCOME_LIABILITIES": "INCOME_LIABILITIES_T",
            "INCOME_ASSETS": "INCOME_ASSETS_T",
            "INCOME_FX": "INCOME_FX_T"
        })
    )

    df = scores.merge(
        income_t,
        how="left",
        on="CONTRAGENTID"
    )

    current_cols = [
        "INCOME_LIABILITIES_T",
        "INCOME_ASSETS_T",
        "INCOME_FX_T"
    ]

    df[current_cols] = df[current_cols].fillna(0)

    # --------------------------------------------------------
    # Liabilities та Assets: потрібен T+1
    # --------------------------------------------------------

    t1_month = shift_month(score_month, 1)
    t1_observed = income_month_available(t1_month)

    if t1_observed:
        income_t1 = (
            read_income(t1_month)
            [["CONTRAGENTID",
              "INCOME_LIABILITIES",
              "INCOME_ASSETS"]]
            .rename(columns={
                "INCOME_LIABILITIES":
                    "INCOME_LIABILITIES_T1",
                "INCOME_ASSETS":
                    "INCOME_ASSETS_T1"
            })
        )

        df = df.merge(
            income_t1,
            how="left",
            on="CONTRAGENTID"
        )

        df[
            [
                "INCOME_LIABILITIES_T1",
                "INCOME_ASSETS_T1"
            ]
        ] = df[
            [
                "INCOME_LIABILITIES_T1",
                "INCOME_ASSETS_T1"
            ]
        ].fillna(0)

        df["NEW_LIABILITIES"] = (
            (df["INCOME_LIABILITIES_T"] <= 0)
            & (df["INCOME_LIABILITIES_T1"] > 0)
        ).astype("boolean")

        df["NEW_ASSETS"] = (
            (df["INCOME_ASSETS_T"] <= 0)
            & (df["INCOME_ASSETS_T1"] > 0)
        ).astype("boolean")

    else:
        df["INCOME_LIABILITIES_T1"] = np.nan
        df["INCOME_ASSETS_T1"] = np.nan

        df["NEW_LIABILITIES"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="boolean"
        )

        df["NEW_ASSETS"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="boolean"
        )

    # --------------------------------------------------------
    # FX: потрібні всі місяці T+1 ... T+5
    # --------------------------------------------------------

    fx_future_months = [
        shift_month(score_month, i)
        for i in range(1, 6)
    ]

    fx_observed = all(
        income_month_available(month)
        for month in fx_future_months
    )

    fx_future_cols = [
        f"INCOME_FX_T{i}"
        for i in range(1, 6)
    ]

    if fx_observed:
        for i, month in enumerate(
            fx_future_months,
            start=1
        ):
            col = f"INCOME_FX_T{i}"

            fx_income = (
                read_income(month)
                [["CONTRAGENTID", "INCOME_FX"]]
                .rename(columns={"INCOME_FX": col})
            )

            df = df.merge(
                fx_income,
                how="left",
                on="CONTRAGENTID"
            )

        df[fx_future_cols] = (
            df[fx_future_cols].fillna(0)
        )

        df["NEW_FX"] = (
            (df["INCOME_FX_T"] <= 0)
            & df[fx_future_cols].gt(0).any(axis=1)
        ).astype("boolean")

    else:
        for col in fx_future_cols:
            df[col] = np.nan

        # Майбутнє ще невідоме — це не False
        df["NEW_FX"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="boolean"
        )

    # --------------------------------------------------------
    # Повний multi-label таргет
    # --------------------------------------------------------

    df["LIABS_ASSETS_LABEL_OBSERVED"] = t1_observed
    df["FX_LABEL_OBSERVED"] = fx_observed

    df["LABELS_FULLY_OBSERVED"] = (
        t1_observed and fx_observed
    )

    df["actual_product"] = pd.NA
    df["n_actual_products"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    if t1_observed and fx_observed:
        product_flags = {
            "Liabilities": "NEW_LIABILITIES",
            "Assets": "NEW_ASSETS",
            "FX": "NEW_FX"
        }

        df["actual_product"] = df.apply(
            lambda row: ", ".join(
                product
                for product, flag
                in product_flags.items()
                if row[flag]
            ) or "None",
            axis=1
        )

        df["n_actual_products"] = (
            df[list(product_flags.values())]
            .sum(axis=1)
            .astype("Int64")
        )

    df["score_month"] = score_month
    df["income_month"] = t1_month
    df["fx_horizon_end"] = fx_future_months[-1]

    return df



