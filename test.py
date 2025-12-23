import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_preprocessor(df: pd.DataFrame):

    # -----------------------------
    # DROP ID + LEAKAGE
    # -----------------------------
    drop_cols = [
        "CONTRAGENTID",
        "IDENTIFYCODE",
        "OTP_AV_LIAB_2025_11",
        "OTP_TURN_2025_11",
        "RATIO_OTP_OTH_2025_11",
        "LOANS_LIMIT",
        "LOANS_AVG",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

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

    if "MONTHLY_INCOME" in df.columns:
        df["MONTHLY_INCOME"] = df["MONTHLY_INCOME"].clip(lower=0)
        df["log_monthly_income"] = np.log1p(df["MONTHLY_INCOME"])
        df.drop(columns="MONTHLY_INCOME", inplace=True)

    if "OTH_BANKS_TURN_2025_11" in df.columns:
        df["log_oth_banks_turn"] = np.log1p(df["OTH_BANKS_TURN_2025_11"])
        df.drop(columns="OTH_BANKS_TURN_2025_11", inplace=True)


    asset_bins = [
        "0_ASSETS", "0-5M_ASSETS", "5M-10M_ASSETS",
        "10M-20M_ASSETS", "15M-30M_ASSETS",
        "20M-30M_ASSETS", ">30M_ASSETS"
    ]
    df.drop(columns=[c for c in asset_bins if c in df.columns], inplace=True)

    categorical_ordinal = ["SEGMENT"]
    categorical_ohe = ["FIRM_KVED"]

    categorical_ordinal = [c for c in categorical_ordinal if c in df.columns]
    categorical_ohe = [c for c in categorical_ohe if c in df.columns]

    numeric_features = [
        c for c in df.columns
        if c not in categorical_ordinal + categorical_ohe
    ]

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        min_frequency=10  # 🔑 захист від explosion
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", ordinal_encoder, categorical_ordinal),
            ("ohe", onehot_encoder, categorical_ohe),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop"
    )

    return df, preprocessor

X_raw, preprocessor = build_preprocessor(df)

X = preprocessor.fit_transform(X_raw)
