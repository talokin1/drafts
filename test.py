a_side = (
    trxs
    .loc[trxs["CONTRAGENTAIDENTIFYCODE"].isin(client_ids),
         ["CONTRAGENTAIDENTIFYCODE", "BANKAID"]]
    .rename(columns={
        "CONTRAGENTAIDENTIFYCODE": "IDENTIFYCODE",
        "BANKAID": "BANKID"
    })
)

b_side = (
    trxs
    .loc[trxs["CONTRAGENTBIDENTIFYCODE"].isin(client_ids),
         ["CONTRAGENTBIDENTIFYCODE", "BANKBID"]]
    .rename(columns={
        "CONTRAGENTBIDENTIFYCODE": "IDENTIFYCODE",
        "BANKBID": "BANKID"
    })
)


client_banks = (
    pd.concat([a_side, b_side], ignore_index=True)
    .dropna(subset=["BANKID"])
    .groupby("IDENTIFYCODE", as_index=False)
    .agg(
        BANKS=("BANKID", lambda x: sorted(set(x)))
    )
)
