import pandas as pd
import numpy as np

df = df.copy()

# вважаємо непорожні значення
df["filled_fields_cnt"] = df.notna().sum(axis=1)

important_cols = [
    "FULL_FIRM_NAME", "OPF", "STATUS", "REGISTRATION_DATE",
    "MSB_SCORE", "MSB_LEVEL", "MSB_SCORE_DATE",
    "KVED", "KVED_DESCR"
]

founder_cols = [c for c in df.columns if c.startswith("FOUNDER_NAME")]
benef_cols = [c for c in df.columns if c.startswith("BENEFICIARY_NAME")]
auth_cols = [c for c in df.columns if c.startswith("AUTHORISED_NAME")]

df["info_score"] = 0

# базові поля
df["info_score"] += df[important_cols].notna().sum(axis=1) * 2

# структурні блоки
df["info_score"] += df[founder_cols].notna().sum(axis=1) * 3
df["info_score"] += df[benef_cols].notna().sum(axis=1) * 4
df["info_score"] += df[auth_cols].notna().sum(axis=1) * 2

sample_size = 500

best_sample = (
    df.sort_values("info_score", ascending=False)
      .head(sample_size)
)

threshold = df["info_score"].quantile(0.95)
best_sample = df[df["info_score"] >= threshold]

best_sample[[
    "info_score",
    "MSB_LEVEL",
    "STATUS"
]].describe(include="all")

best_sample.isna().mean().sort_values()
