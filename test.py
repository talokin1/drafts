def read_scores(month):

    liabs_path = fr"M:\Controlling\Data_Science_Projects\Corp_Liabilities_external_clients\{month}\real_combined_result.csv"

    assets_path = fr"M:\Controlling\Data_Science_Projects\Corp_External_Assets\{month}\model_{month}.parquet"

    fx_path = fr"M:\Controlling\Data_Science_Projects\Corp_External_FX\Results\Models\{month}\fx_external_{month}.parquet"


    liabs = (
        pd.read_csv(liabs_path, dtype={"IDENTIFYCODE": "string"})
        [["IDENTIFYCODE", "PRIMARY"]]
        .rename(columns={"PRIMARY": "LIAB_PRIMARY"})
    )

    liabs = normalize_identifycode(liabs)
    liabs = liabs.drop_duplicates("IDENTIFYCODE")


    assets = (
        pd.read_parquet(assets_path)
        [["IDENTIFYCODE", "PRIMARY"]]
        .rename(columns={"PRIMARY": "ASSETS_PRIMARY"})
    )

    assets = normalize_identifycode(assets)
    assets = assets.drop_duplicates("IDENTIFYCODE")


    if month >= "2026_04" and Path(fx_path).exists():

        fx = (
            pd.read_parquet(fx_path)
            [["IDENTIFYCODE", "PROB_TO_FX"]]
            .rename(columns={"PROB_TO_FX": "FX_PRIMARY"})
        )

        fx = normalize_identifycode(fx)
        fx = fx.drop_duplicates("IDENTIFYCODE")

    else:
        fx = pd.DataFrame({
            "IDENTIFYCODE": pd.Series(dtype="string"),
            "FX_PRIMARY": pd.Series(dtype="float")
        })


    scores = (
        client_map
        .merge(liabs, how="left", on="IDENTIFYCODE")
        .merge(assets, how="left", on="IDENTIFYCODE")
        .merge(fx, how="left", on="IDENTIFYCODE")
    )

    scores = scores.dropna(
        subset=["LIAB_PRIMARY", "ASSETS_PRIMARY", "FX_PRIMARY"],
        how="all"
    )

    return scores