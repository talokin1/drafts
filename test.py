import numpy as np
import pandas as pd

df_adj = df.copy()

# -------------------------
# CONFIG
# -------------------------

RAW_COL = "LIABILITIES_POTENTIAL"
TOTAL_COL = "POTENTIAL_INCOME"
PROP_COL = "PRIMARY"

PRIMARY_THRESHOLD = 0.30

bucket_cols = [
    "0",
    "0-100K",
    "100K-500K",
    "500K-1M",
    "1M-5M",
    "5M-10M",
    "10M+"
]

# -------------------------
# SAVE RAW
# -------------------------

df_adj["LIABILITIES_POTENTIAL_RAW"] = df_adj[RAW_COL]

# -------------------------
# CLEAN
# -------------------------

for col in [RAW_COL, TOTAL_COL, PROP_COL] + bucket_cols:
    df_adj[col] = pd.to_numeric(df_adj[col], errors="coerce").fillna(0)

# -------------------------
# DOMINANT BUCKET
# -------------------------

df_adj["LIABS_TOP_BUCKET"] = df_adj[bucket_cols].idxmax(axis=1)
df_adj["LIABS_TOP_BUCKET_PROB"] = df_adj[bucket_cols].max(axis=1)

# -------------------------
# RECONCILIATION FACTOR
# -------------------------

df_adj["LIABS_RECON_FACTOR"] = 1.0

# 1. PRIMARY нижче порогу, але не катастрофічно:
# просто знижуємо, бо propensity не підтверджує пасиви
df_adj.loc[
    df_adj[PROP_COL] < PRIMARY_THRESHOLD,
    "LIABS_RECON_FACTOR"
] = 0.50

# 2. PRIMARY нижче порогу + домінує bucket 0-100K:
# пасиви можливі, але малий обсяг — сильніше знижуємо
df_adj.loc[
    (df_adj[PROP_COL] < PRIMARY_THRESHOLD)
    & (df_adj["LIABS_TOP_BUCKET"] == "0-100K"),
    "LIABS_RECON_FACTOR"
] = 0.30

# 3. PRIMARY нижче порогу + домінує bucket 0:
# propensity-модель каже, що пасивів не буде — зануляємо
df_adj.loc[
    (df_adj[PROP_COL] < PRIMARY_THRESHOLD)
    & (df_adj["LIABS_TOP_BUCKET"] == "0"),
    "LIABS_RECON_FACTOR"
] = 0.00

# 4. PRIMARY трохи вище порогу, але bucket 0-100K:
# формально позитивний клас, але очікуваний обсяг малий
df_adj.loc[
    (df_adj[PROP_COL] >= PRIMARY_THRESHOLD)
    & (df_adj[PROP_COL] < 0.50)
    & (df_adj["LIABS_TOP_BUCKET"] == "0-100K"),
    "LIABS_RECON_FACTOR"
] = 0.60

# 5. PRIMARY трохи вище порогу, але домінує bucket 0:
# не зануляємо, бо PRIMARY вже пройшов поріг, але знижуємо
df_adj.loc[
    (df_adj[PROP_COL] >= PRIMARY_THRESHOLD)
    & (df_adj[PROP_COL] < 0.50)
    & (df_adj["LIABS_TOP_BUCKET"] == "0"),
    "LIABS_RECON_FACTOR"
] = 0.40

# -------------------------
# APPLY CORRECTION
# -------------------------

df_adj[RAW_COL] = (
    df_adj["LIABILITIES_POTENTIAL_RAW"]
    * df_adj["LIABS_RECON_FACTOR"]
)

# -------------------------
# RECALCULATE TOTAL
# -------------------------

df_adj[TOTAL_COL] = (
    df_adj[TOTAL_COL]
    - df_adj["LIABILITIES_POTENTIAL_RAW"]
    + df_adj[RAW_COL]
)