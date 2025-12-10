df = summary.copy()

df["MFO_list"] = df["MFO"].astype(str).str.split(",\s*")

df = df.explode("MFO_list")
df["MFO_list"] = df["MFO_list"].astype(int)

df = df.merge(
    mfos[["MFO", "NAME"]],
    left_on="MFO_list",
    right_on="MFO",
    how="left"
)

df.rename(columns={"NAME": "bank_name"}, inplace=True)

bank_clients = (
    df.groupby("bank_name")["CLIENT_IDENTIFYCODE"]
      .nunique()
      .reset_index(name="clients")
)

bank_txn = (
    merged.groupby("BANK_USED")   # або BANKID / BANK_USED – твоя колонка з МФО
        .agg(
            n_txn=("SUMMAEQ", "count"),
            total_sum=("SUMMAEQ", "sum")
        )
        .reset_index()
)
bank_txn = bank_txn.merge(
    mfos[["MFO", "NAME"]],
    left_on="BANK_USED",
    right_on="MFO",
    how="left"
).rename(columns={"NAME": "bank_name"})

banks_final = bank_clients.merge(
    bank_txn[["bank_name", "n_txn", "total_sum"]],
    on="bank_name",
    how="left"
)

banks_final = banks_final.sort_values("clients", ascending=False)
banks_final
