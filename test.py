import pandas as pd
from pathlib import Path


def norm_id(s):
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def get_last_scores(files, score_col, new_score_col):
    parts = []

    for file in files:
        month = file.parent.name

        if file.suffix == ".csv":
            df = pd.read_csv(
                file,
                usecols=["IDENTIFYCODE", score_col],
                dtype={"IDENTIFYCODE": "string"}
            )
        else:
            df = pd.read_parquet(
                file,
                columns=["IDENTIFYCODE", score_col]
            )

        df["IDENTIFYCODE"] = norm_id(df["IDENTIFYCODE"])
        df["MONTH"] = month
        parts.append(df)

    result = pd.concat(parts, ignore_index=True)

    # Прибираємо порожні значення ДО вибору останнього місяця
    result = (
        result
        .dropna(subset=[score_col])
        .sort_values("MONTH")
        .drop_duplicates("IDENTIFYCODE", keep="last")
        .rename(columns={
            score_col: new_score_col,
            "MONTH": f"{new_score_col}_MONTH"
        })
    )

    return result


liabs_last = get_last_scores(
    LIABS_BASE.glob("*/real_combined_result.csv"),
    score_col="PRIMARY",
    new_score_col="LIAB_PRIMARY"
)

assets_last = get_last_scores(
    ASSETS_BASE.glob("*/model_*.parquet"),
    score_col="PRIMARY",
    new_score_col="ASSETS_PRIMARY"
)

fx_last = get_last_scores(
    FX_BASE.glob("*/fx_external_*.parquet"),
    score_col="PROB_TO_FX",
    new_score_col="FX_PRIMARY"
)



dataset["IDENTIFYCODE"] = norm_id(dataset["IDENTIFYCODE"])

dataset = (
    dataset
    .merge(liabs_last, on="IDENTIFYCODE", how="left")
    .merge(assets_last, on="IDENTIFYCODE", how="left")
    .merge(fx_last, on="IDENTIFYCODE", how="left")
)