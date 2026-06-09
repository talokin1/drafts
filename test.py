import numpy as np
import pandas as pd

df_adj_fx = df_liabs.copy()

RAW_COL = "FX_POTENTIAL"
TOTAL_COL = "POTENTIAL_INCOME"
PROP_COL = "PROB_TO_FX"

FX_THRESHOLD = 0.01

df_adj_fx["FX_POTENTIAL_RAW"] = df_adj_fx[RAW_COL]

for col in [RAW_COL, TOTAL_COL, PROP_COL]:
    df_adj_fx[col] = pd.to_numeric(df_adj_fx[col], errors="coerce").fillna(0)

df_adj_fx["FX_RECON_FACTOR"] = 1.0

# FX propensity дуже низький — зануляємо FX potential
df_adj_fx.loc[
    df_adj_fx[PROP_COL] < FX_THRESHOLD,
    "FX_RECON_FACTOR"
] = 0.0

# FX propensity слабкий, але не нульовий — можна залишити частину
df_adj_fx.loc[
    (df_adj_fx[PROP_COL] >= FX_THRESHOLD)
    & (df_adj_fx[PROP_COL] < 0.03),
    "FX_RECON_FACTOR"
] = 0.40

# FX propensity нормальний
df_adj_fx.loc[
    (df_adj_fx[PROP_COL] >= 0.03)
    & (df_adj_fx[PROP_COL] < 0.07),
    "FX_RECON_FACTOR"
] = 0.70

# FX propensity високий
df_adj_fx.loc[
    df_adj_fx[PROP_COL] >= 0.07,
    "FX_RECON_FACTOR"
] = 1.0

df_adj_fx[RAW_COL] = (
    df_adj_fx["FX_POTENTIAL_RAW"]
    * df_adj_fx["FX_RECON_FACTOR"]
)

df_adj_fx[TOTAL_COL] = (
    df_adj_fx[TOTAL_COL]
    - df_adj_fx["FX_POTENTIAL_RAW"]
    + df_adj_fx[RAW_COL]
)

df_adj_fx