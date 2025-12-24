df_acq = df[df["is_acquiring"]].copy()
agg_acq = (
    df_acq
    .groupby("CONTRAGENTBIDENTIFYCODE")
    .agg(
        acq_turnover=("SUMMAEQ", "sum"),
        acq_txn_cnt=("SUMMAEQ", "size"),
        sample_platpurpose=("PLATPURPOSE", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
        sample_counterparty=("CONTRAGENTASNAME", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
        reasons=("acq_reason", lambda x: " | ".join(sorted(set(x))))
    )
    .reset_index()
    .sort_values("acq_turnover", ascending=False)
)
