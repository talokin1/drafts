merged["CLIENT_ID"] = merged.apply(
    lambda row: (
        row["CONTRAGENTAIDENTIFYCODE"] 
        if row["TYPE"] == "DEBIT_SELF_ACQ" 
        else row["CONTRAGENTBIDENTIFYCODE"]
    ), 
    axis=1
)


merged["CLIENT_NAME"] = merged.apply(
    lambda row: (
        row["CONTRAGENTASNAME"]
        if row["TYPE"] == "DEBIT_SELF_ACQ"
        else row["CONTRAGENTBSNAME"]
    ),
    axis=1
)


summary = (
    merged.groupby("CLIENT_ID")
    .agg(
        n_txn=("SUMMAEQ", "count"),
        debit_sum=("SUMMAEQ", lambda x: x[merged.loc[x.index, "TYPE"]=="DEBIT_SELF_ACQ"].sum()),
        credit_sum=("SUMMAEQ", lambda x: x[merged.loc[x.index, "TYPE"]=="CREDIT_SELF_ACQ"].sum()),
        total_abs_sum=("SUMMAEQ", lambda x: x.abs().sum()),
        months_active=("PERIOD", "nunique"),
        last_month=("PERIOD", "max"),
        client_name=("CLIENT_NAME", "first"),
        banks_used=("BANKBID", lambda x: ", ".join(sorted(set(x))))
    )
    .reset_index()
    .sort_values("total_abs_sum", ascending=False)
)
