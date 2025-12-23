import pandas as pd
import numpy as np

def preprocess_corp_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # 1. DROP ID COLUMNS
    # -----------------------------
    id_cols = ["CONTRAGENTID", "IDENTIFYCODE"]
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)

    # -----------------------------
    # 2. DROP LEAKAGE FEATURES
    # -----------------------------
    leakage_cols = [
        "OTP_AV_LIAB_2025_11",
        "OTP_TURN_2025_11",
        "RATIO_OTP_OTH_2025_11",
        "LOANS_LIMIT",
        "LOANS_AVG",
    ]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

    # -----------------------------
    # 3. PRODUCT COUNTERS → BINARY
    # -----------------------------
    binary_map = {
        "NB_CARDS_2025_11": "has_cards",
        "NB_LOANS_2025_11": "has_loans",
        "NB_DEPOSITS_2025_11": "has_deposits",
        "NB_SAVINGS_2025_11": "has_savings",
        "NB_ACCOUNTS_2025_11": "has_accounts",
    }

    for src, dst in binary_map.items():
        if src in df.columns:
            df[dst] = (df[src] > 0).astype(int)
            df.drop(columns=src, inplace=True)

    # -----------------------------
    # 4. ASSETS REPRESENTATION
    # -----------------------------
    assets_bins = [
        "0_ASSETS", "0-5M_ASSETS", "5M-10M_ASSETS",
        "10M-20M_ASSETS", "15M-30M_ASSETS",
        "20M-30M_ASSETS", ">30M_ASSETS"
    ]

    for c in assets_bins:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    if "ASSETS_SUIT_AMT_%" in df.columns:
        df["ASSETS_SUIT_AMT_%"] = df["ASSETS_SUIT_AMT_%"].fillna(0)

    # -----------------------------
    # 5. LIABS BINS (KEEP LIMITED SET)
    # -----------------------------
    liabs_bins_drop = [
        "10M+_LIABS",
        ">10M_LIABS",
    ]
    df.drop(columns=[c for c in liabs_bins_drop if c in df.columns], inplace=True)

    # -----------------------------
    # 6. SCALE FEATURES (REQUIRED)
    # -----------------------------
    required_scale = ["total_assets", "total_liabs"]
    for col in required_scale:
        if col not in df.columns:
            raise ValueError(f"Missing required scale feature: {col}")

    df["log_total_assets"] = np.log1p(df["total_assets"])
    df["log_total_liabs"] = np.log1p(df["total_liabs"])

    df.drop(columns=["total_assets", "total_liabs"], inplace=True)

    # -----------------------------
    # 7. LOG TRANSFORMS
    # -----------------------------
    if "MONTHLY_INCOME" in df.columns:
        df["MONTHLY_INCOME"] = df["MONTHLY_INCOME"].clip(lower=0)
        df["log_monthly_income"] = np.log1p(df["MONTHLY_INCOME"])
        df.drop(columns="MONTHLY_INCOME", inplace=True)

    if "OTH_BANKS_TURN_2025_11" in df.columns:
        df["log_oth_banks_turn"] = np.log1p(df["OTH_BANKS_TURN_2025_11"])
        df.drop(columns="OTH_BANKS_TURN_2025_11", inplace=True)

    # -----------------------------
    # 8. CATEGORICAL ENCODING
    # -----------------------------
    cat_cols = ["SEGMENT", "FIRM_KVED"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # -----------------------------
    # 9. FINAL CLEANUP
    # -----------------------------
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df
