target_clusters = [
    "CEO",
    "FORBES NEXT",
    "Forbes 100",
    "Gray Cardinals"
]

df_filt = df[df["Кластер"].isin(target_clusters)]



result = (
    df_filt
    .groupby("Кластер")
    .agg(
        clients_cnt=("CONTRAGENTID", "nunique"),
        avg_total_score=("TOTAL SCORE", "mean")
    )
    .reset_index()
)


result["avg_total_score"] = result["avg_total_score"].round(2)
