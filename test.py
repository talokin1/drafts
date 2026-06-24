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



client_map = clients[
    ["IDENTIFYCODE", "CONTRAGENTID"]
].copy()

client_map = normalize_identifycode(client_map)
client_map = normalize_contragentid(client_map)

client_map = (
    client_map
    .dropna(subset=["IDENTIFYCODE", "CONTRAGENTID"])
    .drop_duplicates("IDENTIFYCODE")
)





def read_income(month):

    path = fr"M:\Controlling\Data_Science_Projects\Income_Data\income_wide_corporate_clients_{month}.csv"

    income = (
        pd.read_csv(
            path,
            dtype={"CONTRAGENTID": "string"}
        )
        .rename(columns={"COM_CORP_FX_FOR_PAY": "INCOME_FX"})
    )

    income = income[
        [
            "CONTRAGENTID",
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ].copy()

    income = normalize_contragentid(income)

    income[
        [
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ] = income[
        [
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ].apply(pd.to_numeric, errors="coerce").fillna(0)

    return (
        income
        .groupby("CONTRAGENTID", as_index=False)
        .sum()
    )




def read_income(month):

    path = fr"M:\Controlling\Data_Science_Projects\Income_Data\income_wide_corporate_clients_{month}.csv"

    income = (
        pd.read_csv(
            path,
            dtype={"CONTRAGENTID": "string"}
        )
        .rename(columns={"COM_CORP_FX_FOR_PAY": "INCOME_FX"})
    )

    income = income[
        [
            "CONTRAGENTID",
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ].copy()

    income = normalize_contragentid(income)

    income[
        [
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ] = income[
        [
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ].apply(pd.to_numeric, errors="coerce").fillna(0)

    return (
        income
        .groupby("CONTRAGENTID", as_index=False)
        .sum()
    )