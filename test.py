df_banks = summary.copy()

# explode by bank
df_banks["bank"] = df_banks["bank_used"].str.split(",\s*")
df_banks = df_banks.explode("bank")

banks_clients = (
    df_banks.groupby("bank")["CLIENT_IDENTIFYCODE"]
    .nunique()
    .reset_index(name="clients")
)

banks_txn = (
    merged.groupby("BANK_USED")
    .agg(
        n_txn=("SUMMAEQ", "count"),
        total_sum=("SUMMAEQ", "sum")
    )
    .reset_index()
)

banks_final = (
    banks_clients
    .merge(banks_txn, left_on="bank", right_on="BANK_USED", how="left")
    .drop(columns=["BANK_USED"])
)
