final_clients = (
    clients[["IDENTIFYCODE"]]
    
    .merge(
        liab_clients[["IDENTIFYCODE", "PRIMARY"]]
        .rename(columns={"PRIMARY": "LIAB_PRIMARY"}),
        how="left",
        on="IDENTIFYCODE"
    )
    
    .merge(
        assets_clients[["IDENTIFYCODE", "PRIMARY"]]
        .rename(columns={"PRIMARY": "ASSETS_PRIMARY"}),
        how="left",
        on="IDENTIFYCODE"
    )
    
    .merge(
        fx_clients[["IDENTIFYCODE", "PROB_TO_FX"]]
        .rename(columns={"PROB_TO_FX": "FX_PRIMARY"}),
        how="left",
        on="IDENTIFYCODE"
    )
)