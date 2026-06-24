from pathlib import Path
import pandas as pd
import numpy as np




client_map = (
    clients[["IDENTIFYCODE", "CONTRAGENTID"]]
    .dropna(subset=["IDENTIFYCODE", "CONTRAGENTID"])
    .drop_duplicates("IDENTIFYCODE")
)


def read_income(month):
    
    path = fr"M:\Controlling\Data_Science_Projects\Income_Data\income_wide_corporate_clients_{month}.csv"
    
    income = (
        pd.read_csv(path)
        .rename(columns={"COM_CORP_FX_FOR_PAY": "INCOME_FX"})
    )
    
    income = income[
        [
            "CONTRAGENTID",
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ].copy()
    
    income[
        [
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ] = income[
        [
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "INCOME_FX"
        ]
    ].fillna(0)
    
    return income.groupby("CONTRAGENTID", as_index=False).sum()



def read_scores(month):
    
    liabs_path = fr"M:\Controlling\Data_Science_Projects\Corp_Liabilities_external_clients\{month}\real_combined_result.csv"
    
    assets_path = fr"M:\Controlling\Data_Science_Projects\Corp_External_Assets\{month}\model_{month}.parquet"
    
    fx_path = fr"M:\Controlling\Data_Science_Projects\Corp_External_FX\Results\Models\{month}\fx_external_{month}.parquet"
    
    
    # Liabilities
    liabs = (
        pd.read_csv(liabs_path)
        [["IDENTIFYCODE", "PRIMARY"]]
        .rename(columns={"PRIMARY": "LIAB_PRIMARY"})
        .drop_duplicates("IDENTIFYCODE")
    )
    
    
    # Assets
    assets = (
        pd.read_parquet(assets_path)
        [["IDENTIFYCODE", "PRIMARY"]]
        .rename(columns={"PRIMARY": "ASSETS_PRIMARY"})
        .drop_duplicates("IDENTIFYCODE")
    )
    
    
    # FX починаємо лише з 2026_04
    if month >= "2026_04" and Path(fx_path).exists():
        
        fx = (
            pd.read_parquet(fx_path)
            [["IDENTIFYCODE", "PROB_TO_FX"]]
            .rename(columns={"PROB_TO_FX": "FX_PRIMARY"})
            .drop_duplicates("IDENTIFYCODE")
        )
        
    else:
        fx = pd.DataFrame(
            columns=["IDENTIFYCODE", "FX_PRIMARY"]
        )
    
    
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


def build_validation_month(score_month, next_income_month):
    
    scores = read_scores(score_month)
    
    income_t = read_income(score_month).rename(
        columns={
            "INCOME_LIABILITIES": "INCOME_LIABILITIES_T",
            "INCOME_ASSETS": "INCOME_ASSETS_T",
            "INCOME_FX": "INCOME_FX_T"
        }
    )
    
    income_t1 = read_income(next_income_month).rename(
        columns={
            "INCOME_LIABILITIES": "INCOME_LIABILITIES_T1",
            "INCOME_ASSETS": "INCOME_ASSETS_T1",
            "INCOME_FX": "INCOME_FX_T1"
        }
    )
    
    df = (
        scores
        .merge(income_t, how="left", on="CONTRAGENTID")
        .merge(income_t1, how="left", on="CONTRAGENTID")
    )
    
    income_cols = [
        "INCOME_LIABILITIES_T",
        "INCOME_ASSETS_T",
        "INCOME_FX_T",
        "INCOME_LIABILITIES_T1",
        "INCOME_ASSETS_T1",
        "INCOME_FX_T1"
    ]
    
    df[income_cols] = df[income_cols].fillna(0)
    
    
    # Нове залучення до пасивів:
    # скор був, у t доходу не було, у t+1 дохід з'явився
    
    df["NEW_LIABILITIES"] = (
        df["LIAB_PRIMARY"].notna()
        & (df["INCOME_LIABILITIES_T"] <= 0)
        & (df["INCOME_LIABILITIES_T1"] > 0)
    )
    
    df["NEW_ASSETS"] = (
        df["ASSETS_PRIMARY"].notna()
        & (df["INCOME_ASSETS_T"] <= 0)
        & (df["INCOME_ASSETS_T1"] > 0)
    )
    
    df["NEW_FX"] = (
        df["FX_PRIMARY"].notna()
        & (df["INCOME_FX_T"] <= 0)
        & (df["INCOME_FX_T1"] > 0)
    )
    
    
    df["n_actual_products"] = (
        df[["NEW_LIABILITIES", "NEW_ASSETS", "NEW_FX"]]
        .sum(axis=1)
    )
    
    
    # Якщо одночасно активувалося кілька продуктів —
    # беремо продукт з найбільшим фактичним доходом у t+1
    
    df["actual_product"] = np.select(
        [
            df["NEW_LIABILITIES"],
            df["NEW_ASSETS"],
            df["NEW_FX"]
        ],
        [
            "Liabilities",
            "Assets",
            "FX"
        ],
        default=np.nan
    )
    
    
    # Для клієнтів з кількома продуктами визначаємо найбільший
    multiple_mask = df["n_actual_products"] > 1
    
    income_t1_cols = {
        "Liabilities": "INCOME_LIABILITIES_T1",
        "Assets": "INCOME_ASSETS_T1",
        "FX": "INCOME_FX_T1"
    }
    
    for idx in df[multiple_mask].index:
        
        available_products = []
        
        if df.loc[idx, "NEW_LIABILITIES"]:
            available_products.append("Liabilities")
            
        if df.loc[idx, "NEW_ASSETS"]:
            available_products.append("Assets")
            
        if df.loc[idx, "NEW_FX"]:
            available_products.append("FX")
        
        df.loc[idx, "actual_product"] = max(
            available_products,
            key=lambda x: df.loc[idx, income_t1_cols[x]]
        )
    
    
    df["score_month"] = score_month
    df["income_month"] = next_income_month
    
    return df[
        [
            "score_month",
            "income_month",
            "IDENTIFYCODE",
            "CONTRAGENTID",
            "LIAB_PRIMARY",
            "ASSETS_PRIMARY",
            "FX_PRIMARY",
            "INCOME_LIABILITIES_T",
            "INCOME_ASSETS_T",
            "INCOME_FX_T",
            "INCOME_LIABILITIES_T1",
            "INCOME_ASSETS_T1",
            "INCOME_FX_T1",
            "NEW_LIABILITIES",
            "NEW_ASSETS",
            "NEW_FX",
            "n_actual_products",
            "actual_product"
        ]
    ]


month_pairs = [
    ("2025_05", "2025_06"),
    ("2025_06", "2025_07"),
    ("2025_07", "2025_08"),
    ("2025_08", "2025_09"),
    ("2025_09", "2025_10"),
    ("2025_10", "2025_11"),
    ("2025_11", "2025_12"),
    ("2025_12", "2026_01"),
    ("2026_01", "2026_02"),
    ("2026_02", "2026_03"),
    ("2026_03", "2026_04"),
    ("2026_04", "2026_05"),
]



validation_parts = []

for score_month, income_month in month_pairs:
    
    try:
        month_df = build_validation_month(
            score_month=score_month,
            next_income_month=income_month
        )
        
        validation_parts.append(month_df)
        print(f"{score_month}: OK, rows = {len(month_df):,}")
        
    except Exception as e:
        print(f"{score_month}: ERROR -> {e}")


validation_dataset = pd.concat(
    validation_parts,
    ignore_index=True
)

validation_dataset = validation_dataset[
    validation_dataset["actual_product"].notna()
].copy()

validation_dataset.shape









validation_dataset_clean = validation_dataset[
    validation_dataset["n_actual_products"] == 1
].copy()

validation_dataset_clean.shape