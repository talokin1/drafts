import numpy as np
import pandas as pd

df_adj_assets = df_liabs.copy()

RAW_COL = "ASSETS_POTENTIAL"
TOTAL_COL = "POTENTIAL_INCOME"
PROP_COL = "PRIMARY_ASSETS"

PRIMARY_THRESHOLD = 0.30

bucket_cols = [
    "0_ASSETS",
    "0-5M_ASSETS",
    "5M-10M_ASSETS",
    "10M-20M_ASSETS",
    "20M-30M_ASSETS",
    ">30M_ASSETS"
]

df_adj_assets["ASSETS_POTENTIAL_RAW"] = df_adj_assets[RAW_COL]

for col in [RAW_COL, TOTAL_COL, PROP_COL] + bucket_cols:
    df_adj_assets[col] = pd.to_numeric(df_adj_assets[col], errors="coerce").fillna(0)

df_adj_assets["ASSETS_TOP_BUCKET"] = df_adj_assets[bucket_cols].idxmax(axis=1)
df_adj_assets["ASSETS_TOP_BUCKET_PROB"] = df_adj_assets[bucket_cols].max(axis=1)

df_adj_assets["ASSETS_RECON_FACTOR"] = 1.0

# PRIMARY нижче порогу — загально знижуємо
df_adj_assets.loc[
    df_adj_assets[PROP_COL] < PRIMARY_THRESHOLD,
    "ASSETS_RECON_FACTOR"
] = 0.50

# PRIMARY нижче порогу + top bucket = 0-5M — активи можливі, але малий обсяг
df_adj_assets.loc[
    (df_adj_assets[PROP_COL] < PRIMARY_THRESHOLD)
    & (df_adj_assets["ASSETS_TOP_BUCKET"] == "0-5M_ASSETS"),
    "ASSETS_RECON_FACTOR"
] = 0.30

# PRIMARY нижче порогу + top bucket = 0 — активів не очікується
df_adj_assets.loc[
    (df_adj_assets[PROP_COL] < PRIMARY_THRESHOLD)
    & (df_adj_assets["ASSETS_TOP_BUCKET"] == "0_ASSETS"),
    "ASSETS_RECON_FACTOR"
] = 0.00

# PRIMARY 0.30-0.50 + top bucket = 0-5M — слабкий, але позитивний сигнал
df_adj_assets.loc[
    (df_adj_assets[PROP_COL] >= PRIMARY_THRESHOLD)
    & (df_adj_assets[PROP_COL] < 0.50)
    & (df_adj_assets["ASSETS_TOP_BUCKET"] == "0-5M_ASSETS"),
    "ASSETS_RECON_FACTOR"
] = 0.60

# PRIMARY 0.30-0.50 + top bucket = 0 — не зануляємо, але знижуємо
df_adj_assets.loc[
    (df_adj_assets[PROP_COL] >= PRIMARY_THRESHOLD)
    & (df_adj_assets[PROP_COL] < 0.50)
    & (df_adj_assets["ASSETS_TOP_BUCKET"] == "0_ASSETS"),
    "ASSETS_RECON_FACTOR"
] = 0.40

df_adj_assets[RAW_COL] = (
    df_adj_assets["ASSETS_POTENTIAL_RAW"]
    * df_adj_assets["ASSETS_RECON_FACTOR"]
)

df_adj_assets[TOTAL_COL] = (
    df_adj_assets[TOTAL_COL]
    - df_adj_assets["ASSETS_POTENTIAL_RAW"]
    + df_adj_assets[RAW_COL]
)

df_adj_assets