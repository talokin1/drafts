import numpy as np
import pandas as pd

df_adj = df.copy()

# =========================
# CONFIG
# =========================

RAW_COL = "LIABILITIES_POTENTIAL"
PROP_COL = "PRIMARY"

BUCKET_COLS = [
    "0",
    "0-100K",
    "100K-500K",
    "500K-1M",
    "1M-5M",
    "5M-10M",
    "10M+"
]

THRESHOLD = 0.30  # optimal F1 threshold for PRIMARY


# =========================
# CLEAN
# =========================

cols_to_num = [RAW_COL, PROP_COL] + BUCKET_COLS

for col in cols_to_num:
    df_adj[col] = pd.to_numeric(df_adj[col], errors="coerce").fillna(0)

df_adj[RAW_COL] = df_adj[RAW_COL].clip(lower=0)

for col in [PROP_COL] + BUCKET_COLS:
    df_adj[col] = df_adj[col].clip(0, 1)


# =========================
# BASIC SIGNALS
# =========================

df_adj["LIABS_PRIMARY_FLAG"] = (df_adj[PROP_COL] >= THRESHOLD).astype(int)

df_adj["LIABS_ZERO_PROB"] = df_adj["0"]

df_adj["LIABS_TOP_BUCKET"] = df_adj[BUCKET_COLS].idxmax(axis=1)

df_adj["LIABS_TOP_BUCKET_PROB"] = df_adj[BUCKET_COLS].max(axis=1)


# =========================
# RECONCILIATION FACTOR
# =========================

df_adj["LIABS_RECON_FACTOR"] = 1.0

# 1) PRIMARY нижче порогу — сильно знижуємо потенціал
df_adj.loc[
    df_adj[PROP_COL] < THRESHOLD,
    "LIABS_RECON_FACTOR"
] = 0.25

# 2) PRIMARY нижче порогу + bucket 0 дуже ймовірний — майже зануляємо
df_adj.loc[
    (df_adj[PROP_COL] < THRESHOLD) & (df_adj["LIABS_ZERO_PROB"] >= 0.60),
    "LIABS_RECON_FACTOR"
] = 0.05

# 3) PRIMARY вище порогу, але bucket 0 високий — мʼяко дисконтуємо
df_adj.loc[
    (df_adj[PROP_COL] >= THRESHOLD) & (df_adj["LIABS_ZERO_PROB"] >= 0.60),
    "LIABS_RECON_FACTOR"
] = 0.50


# =========================
# ADJUSTED LIABILITIES POTENTIAL
# =========================

df_adj["LIABILITIES_POTENTIAL_RAW"] = df_adj[RAW_COL]

df_adj["LIABILITIES_POTENTIAL_ADJ"] = (
    df_adj["LIABILITIES_POTENTIAL_RAW"] * df_adj["LIABS_RECON_FACTOR"]
)


# =========================
# RECALCULATE TOTAL POTENTIAL
# =========================

df_adj["POTENTIAL_INCOME_ADJ_LIABS"] = (
    df_adj["POTENTIAL_INCOME"]
    - df_adj["LIABILITIES_POTENTIAL_RAW"]
    + df_adj["LIABILITIES_POTENTIAL_ADJ"]
)


# =========================
# BUSINESS STATUS
# =========================

df_adj["LIABS_STATUS"] = np.select(
    [
        (df_adj[PROP_COL] >= THRESHOLD) & (df_adj["LIABS_ZERO_PROB"] < 0.60),
        (df_adj[PROP_COL] >= THRESHOLD) & (df_adj["LIABS_ZERO_PROB"] >= 0.60),
        (df_adj[PROP_COL] < THRESHOLD) & (df_adj["LIABS_ZERO_PROB"] < 0.60),
        (df_adj[PROP_COL] < THRESHOLD) & (df_adj["LIABS_ZERO_PROB"] >= 0.60),
    ],
    [
        "Пасиви підтверджені propensity-моделлю",
        "PRIMARY підтверджує, але bucket=0 високий",
        "PRIMARY нижче порогу — потенціал знижено",
        "PRIMARY нижче порогу і bucket=0 високий — потенціал майже занулено",
    ],
    default="Нейтральний кейс"
)


# =========================
# RESULT FOR REVIEW
# =========================

liabs_review_cols = [
    "IDENTIFYCODE",
    "FIRM_NAME",
    "FIRM_TYPE",
    "POTENTIAL_INCOME",
    "POTENTIAL_INCOME_ADJ_LIABS",
    "LIABILITIES_POTENTIAL_RAW",
    "LIABILITIES_POTENTIAL_ADJ",
    "PRIMARY",
    "LIABS_PRIMARY_FLAG",
    "LIABS_ZERO_PROB",
    "LIABS_TOP_BUCKET",
    "LIABS_TOP_BUCKET_PROB",
    "LIABS_RECON_FACTOR",
    "LIABS_STATUS"
]

liabs_review_cols = [c for c in liabs_review_cols if c in df_adj.columns]

df_liabs_review = df_adj[liabs_review_cols].copy()

df_liabs_review.head()