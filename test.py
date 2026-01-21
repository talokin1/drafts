cluster_block_summary = (
    final_summary
    .groupby("Кластер", as_index=False)
    .agg(
        clients_cnt = ("CONTRAGENTID", "count"),
        avg_delta_total = ("DELTA_TOTAL_SCORE", "mean"),
        avg_delta_business = ("DELTA_BUSINESS_SCORE", "mean"),
        avg_delta_portfolio = ("DELTA_PORTFOLIO_SCORE", "mean"),
        avg_delta_daily = ("DELTA_DAILY_SCORE", "mean"),
    )
)

for col in ["business", "portfolio", "daily"]:
    cluster_block_summary[f"share_{col}"] = (
        cluster_block_summary[f"avg_delta_{col}"]
        / cluster_block_summary["avg_delta_total"]
    )


cluster_main_driver = (
    final_summary
    .groupby(["Кластер", "MAIN_GROWTH_DRIVER"])
    .size()
    .reset_index(name="clients_cnt")
)
cluster_main_driver["share"] = (
    cluster_main_driver
    .groupby("Кластер")["clients_cnt"]
    .transform(lambda x: x / x.sum())
)


def explode_drivers(df, cluster_col, drivers_col, block_name):
    tmp = (
        df[[cluster_col, drivers_col]]
        .dropna()
        .explode(drivers_col)
    )
    tmp["BLOCK"] = block_name
    tmp.rename(columns={drivers_col: "FEATURE"}, inplace=True)
    return tmp
drivers_long = pd.concat([
    explode_drivers(final_summary, "Кластер", "TOP_BUSINESS_DRIVERS", "BUSINESS"),
    explode_drivers(final_summary, "Кластер", "TOP_PORTFOLIO_DRIVERS", "PORTFOLIO"),
    explode_drivers(final_summary, "Кластер", "TOP_DAILY_DRIVERS", "DAILY"),
])
cluster_feature_impact = (
    drivers_long
    .groupby(["Кластер", "BLOCK", "FEATURE"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["Кластер", "BLOCK", "cnt"], ascending=[True, True, False])
)
