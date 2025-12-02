# ID мерчанта
merged["MERCHANT_ID"] = merged.apply(
    lambda row: row["CONTRAGENTAID"] if row["TYPE"] == "DEBIT_SELF_ACQ" else row["CONTRAGENTBID"],
    axis=1
)

# Назва мерчанта
merged["MERCHANT_NAME"] = merged.apply(
    lambda row: row["CONTRAGENTASNAME"] if row["TYPE"] == "DEBIT_SELF_ACQ" else row["CONTRAGENTBSNAME"],
    axis=1
)

# Ідентифікаційний код мерчанта
merged["MERCHANT_IDENTIFYCODE"] = merged.apply(
    lambda row: row["CONTRAGENTAIDENTIFYCODE"] if row["TYPE"] == "DEBIT_SELF_ACQ"
    else row["CONTRAGENTBIDENTIFYCODE"],
    axis=1
)

# Банк мерчанта
merged["MERCHANT_BANK"] = merged.apply(
    lambda row: row["BANKBID"] if row["TYPE"] == "DEBIT_SELF_ACQ" else row["BANKAID"],
    axis=1
)


summary = (
    merged.groupby(["MERCHANT_IDENTIFYCODE", "TYPE"])
    .agg(
        n_txn=("SUMMAEQ", "count"),
        total_sum=("SUMMAEQ", "sum"),
        months_active=("PERIOD", "nunique"),
        last_month=("PERIOD", "max"),
        MERCHANT_NAME=("MERCHANT_NAME", "first"),
        MERCHANT_ID=("MERCHANT_ID", "first"),
        MERCHANT_BANK=("MERCHANT_BANK", lambda x: ", ".join(sorted(set(str(v) for v in x))))
    )
    .reset_index()
    .sort_values("total_sum", ascending=False)
)

