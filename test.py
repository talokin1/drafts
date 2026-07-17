test = clients[["IDENTIFYCODE"]].merge(
    liabs[["IDENTIFYCODE", "PRIMARY"]],
    how="left",
    on="IDENTIFYCODE"
)
liab_clients = test[test["PRIMARY"].notna()]


test = clients[["IDENTIFYCODE"]].merge(
    assets[["IDENTIFYCODE", "PRIMARY"]],
    how="left",
    on="IDENTIFYCODE"
)
assets_clients = test[test["PRIMARY"].notna()]


test = clients[["IDENTIFYCODE"]].merge(
    fx[["IDENTIFYCODE", "PROB_TO_FX"]],
    how="left",
    on="IDENTIFYCODE"
)
fx_clients = test[test["PROB_TO_FX"].notna()]