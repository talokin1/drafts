import numpy as np
import pandas as pd

df_adj = df.copy()

# 1. Зберігаємо старий прогноз
df_adj["LIABILITIES_POTENTIAL_RAW"] = df_adj["LIABILITIES_POTENTIAL"]

# 2. Колонки bucket-моделі
bucket_values = {
    "0": 0,
    "0-100K": 50_000,
    "100K-500K": 300_000,
    "500K-1M": 750_000,
    "1M-5M": 3_000_000,
    "5M-10M": 7_500_000,
    "10M+": 12_000_000
}

bucket_cols = list(bucket_values.keys())

# 3. Чистимо числа
cols = ["LIABILITIES_POTENTIAL", "POTENTIAL_INCOME", "PRIMARY"] + bucket_cols

for col in cols:
    df_adj[col] = pd.to_numeric(df_adj[col], errors="coerce").fillna(0)

# 4. Рахуємо expected volume по всіх bucket
df_adj["LIABS_EXPECTED_VOLUME"] = sum(
    df_adj[col] * value for col, value in bucket_values.items()
)

# 5. Фактор по PRIMARY
df_adj["LIABS_PRIMARY_FACTOR"] = np.select(
    [
        df_adj["PRIMARY"] < 0.30,
        df_adj["PRIMARY"].between(0.30, 0.50, inclusive="left"),
        df_adj["PRIMARY"].between(0.50, 0.70, inclusive="left"),
        df_adj["PRIMARY"] >= 0.70
    ],
    [
        0.20,   # propensity не підтверджує
        0.50,   # слабке підтвердження
        0.80,   # нормальне підтвердження
        1.00    # сильне підтвердження
    ],
    default=0.20
)

# 6. Фактор по очікуваному обсягу
df_adj["LIABS_VOLUME_FACTOR"] = np.select(
    [
        df_adj["LIABS_EXPECTED_VOLUME"] < 100_000,
        df_adj["LIABS_EXPECTED_VOLUME"].between(100_000, 500_000, inclusive="left"),
        df_adj["LIABS_EXPECTED_VOLUME"].between(500_000, 1_000_000, inclusive="left"),
        df_adj["LIABS_EXPECTED_VOLUME"] >= 1_000_000
    ],
    [
        0.25,
        0.50,
        0.75,
        1.00
    ],
    default=0.25
)

# 7. Загальний reconciliation factor
df_adj["LIABS_RECON_FACTOR"] = (
    df_adj["LIABS_PRIMARY_FACTOR"] * df_adj["LIABS_VOLUME_FACTOR"]
)

# 8. Скоригований liabilities potential
df_adj["LIABILITIES_POTENTIAL"] = (
    df_adj["LIABILITIES_POTENTIAL_RAW"] * df_adj["LIABS_RECON_FACTOR"]
)

# 9. Перерахунок total potential
df_adj["POTENTIAL_INCOME"] = (
    df_adj["POTENTIAL_INCOME"]
    - df_adj["LIABILITIES_POTENTIAL_RAW"]
    + df_adj["LIABILITIES_POTENTIAL"]
)