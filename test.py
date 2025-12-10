type_stats = (
    merged.groupby("TYPE")
    .agg(
        clients=("CLIENT_IDENTIFYCODE", "nunique"),
        total_sum=("SUMMAEQ", "sum")
    )
    .reset_index()
)
type_stats.loc["TOTAL"] = [
    "TOTAL",
    summary["CLIENT_IDENTIFYCODE"].nunique(),
    merged["SUMMAEQ"].sum()
]
