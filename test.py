import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================

ID_COL = "IDENTIFYCODE"

RAW_LIAB_POTENTIAL_COL = "LIABILITIES_POTENTIAL"

# Колонка з основною propensity probability.
# Якщо у тебе PRIMARY означає ймовірність "залучення пасивів",
# залишаємо так.
PROPENSITY_COL = "PRIMARY"

# Bucket probability columns з propensity-моделі
BUCKET_COLS = [
    "0",
    "0-100K",
    "100K-500K",
    "500K-1M",
    "1M-5M",
    "5M-10M",
    "10M+"
]

# Умовні representative values для кожного bucket.
# Їх можна буде потім уточнити з бізнесом або по історичних медіанах.
BUCKET_VALUES = {
    "0": 0,
    "0-100K": 50_000,
    "100K-500K": 300_000,
    "500K-1M": 750_000,
    "1M-5M": 3_000_000,
    "5M-10M": 7_500_000,
    "10M+": 12_000_000
}

# Обсяг, відносно якого нормалізуємо expected volume.
# Наприклад, якщо expected volume = 3M, то bucket_score буде близько 1.
# Якщо хочеш більш агресивно піднімати великих клієнтів — можна зменшити до 1_000_000.
VOLUME_NORMALIZATION_VALUE = 3_000_000

# Мінімальний множник, щоб не вбивати потенціал повністю
MIN_PROPENSITY_FACTOR = 0.05
MIN_BUCKET_FACTOR = 0.10

# Стеля для множників
MAX_PROPENSITY_FACTOR = 1.00
MAX_BUCKET_FACTOR = 1.00

# Якщо ймовірність bucket=0 дуже висока, додатково штрафуємо
ZERO_BUCKET_SOFT_THRESHOLD = 0.50
ZERO_BUCKET_HARD_THRESHOLD = 0.80

# Якщо propensity дуже низький і bucket=0 дуже високий,
# тоді можемо майже занулити прогноз.
VERY_LOW_PROPENSITY_THRESHOLD = 0.10
VERY_HIGH_ZERO_BUCKET_THRESHOLD = 0.80


# ============================================================
# 2. COPY DATA
# ============================================================

df_rec = df.copy()

# Нормалізуємо назви колонок, якщо випадково є пробіли
df_rec.columns = df_rec.columns.astype(str).str.strip()


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_cols = [ID_COL, RAW_LIAB_POTENTIAL_COL, PROPENSITY_COL] + BUCKET_COLS

missing_cols = [col for col in required_cols if col not in df_rec.columns]

if missing_cols:
    raise ValueError(
        f"У df немає потрібних колонок: {missing_cols}\n"
        f"Наявні колонки: {df_rec.columns.tolist()}"
    )


# ============================================================
# 4. CLEAN NUMERIC COLUMNS
# ============================================================

numeric_cols = [RAW_LIAB_POTENTIAL_COL, PROPENSITY_COL] + BUCKET_COLS

for col in numeric_cols:
    df_rec[col] = pd.to_numeric(df_rec[col], errors="coerce")

# Якщо десь NaN — замінюємо на 0
df_rec[numeric_cols] = df_rec[numeric_cols].fillna(0)

# Потенціал не може бути менше 0
df_rec[RAW_LIAB_POTENTIAL_COL] = df_rec[RAW_LIAB_POTENTIAL_COL].clip(lower=0)

# Ймовірності обрізаємо в межах [0, 1]
for col in [PROPENSITY_COL] + BUCKET_COLS:
    df_rec[col] = df_rec[col].clip(lower=0, upper=1)


# ============================================================
# 5. OPTIONAL: NORMALIZE BUCKET PROBABILITIES
# ============================================================

# Іноді bucket probabilities можуть не сумуватися рівно в 1 через округлення
# або особливості експорту. Нормалізуємо їх.
df_rec["LIABS_BUCKET_PROB_SUM"] = df_rec[BUCKET_COLS].sum(axis=1)

for col in BUCKET_COLS:
    df_rec[col] = np.where(
        df_rec["LIABS_BUCKET_PROB_SUM"] > 0,
        df_rec[col] / df_rec["LIABS_BUCKET_PROB_SUM"],
        df_rec[col]
    )

df_rec["LIABS_BUCKET_PROB_SUM_AFTER_NORM"] = df_rec[BUCKET_COLS].sum(axis=1)


# ============================================================
# 6. EXPECTED LIABILITIES VOLUME
# ============================================================

df_rec["LIABS_EXPECTED_VOLUME"] = 0.0

for bucket_col, bucket_value in BUCKET_VALUES.items():
    df_rec["LIABS_EXPECTED_VOLUME"] += df_rec[bucket_col] * bucket_value


# ============================================================
# 7. BUCKET SCORE
# ============================================================

# Перетворюємо expected volume у коефіцієнт [MIN_BUCKET_FACTOR; 1].
# Чим більший очікуваний обсяг — тим менше штрафуємо potential.
df_rec["LIABS_BUCKET_FACTOR_RAW"] = (
    df_rec["LIABS_EXPECTED_VOLUME"] / VOLUME_NORMALIZATION_VALUE
)

df_rec["LIABS_BUCKET_FACTOR"] = df_rec["LIABS_BUCKET_FACTOR_RAW"].clip(
    lower=MIN_BUCKET_FACTOR,
    upper=MAX_BUCKET_FACTOR
)


# ============================================================
# 8. PROPENSITY FACTOR
# ============================================================

# Базово беремо PRIMARY як probability.
# Але не даємо йому стати рівно 0, щоб модель не зануляла все надто агресивно.
df_rec["LIABS_PROPENSITY_FACTOR"] = df_rec[PROPENSITY_COL].clip(
    lower=MIN_PROPENSITY_FACTOR,
    upper=MAX_PROPENSITY_FACTOR
)


# ============================================================
# 9. ZERO-BUCKET PENALTY
# ============================================================

# Якщо P(bucket=0) висока, значить модель пасивів каже:
# "скоріше за все, клієнт не принесе пасивів".
# Тому додатково знижуємо potential.
df_rec["LIABS_ZERO_BUCKET_PROB"] = df_rec["0"]

df_rec["LIABS_ZERO_BUCKET_PENALTY"] = 1.0

# Мʼякий штраф
df_rec.loc[
    df_rec["LIABS_ZERO_BUCKET_PROB"] >= ZERO_BUCKET_SOFT_THRESHOLD,
    "LIABS_ZERO_BUCKET_PENALTY"
] = 0.5

# Жорсткіший штраф
df_rec.loc[
    df_rec["LIABS_ZERO_BUCKET_PROB"] >= ZERO_BUCKET_HARD_THRESHOLD,
    "LIABS_ZERO_BUCKET_PENALTY"
] = 0.2


# ============================================================
# 10. FINAL ADJUSTED LIABILITIES POTENTIAL
# ============================================================

df_rec["LIABILITIES_POTENTIAL_RAW"] = df_rec[RAW_LIAB_POTENTIAL_COL]

df_rec["LIABILITIES_POTENTIAL_ADJ"] = (
    df_rec["LIABILITIES_POTENTIAL_RAW"]
    * df_rec["LIABS_PROPENSITY_FACTOR"]
    * df_rec["LIABS_BUCKET_FACTOR"]
    * df_rec["LIABS_ZERO_BUCKET_PENALTY"]
)


# ============================================================
# 11. EXTRA HARD RULE FOR OBVIOUSLY BAD CASES
# ============================================================

# Якщо і propensity дуже низький, і P(0) дуже висока,
# тоді майже зануляємо potential.
bad_liabs_mask = (
    (df_rec[PROPENSITY_COL] < VERY_LOW_PROPENSITY_THRESHOLD)
    & (df_rec["LIABS_ZERO_BUCKET_PROB"] >= VERY_HIGH_ZERO_BUCKET_THRESHOLD)
)

df_rec.loc[bad_liabs_mask, "LIABILITIES_POTENTIAL_ADJ"] = (
    df_rec.loc[bad_liabs_mask, "LIABILITIES_POTENTIAL_RAW"] * 0.02
)


# ============================================================
# 12. MOST LIKELY BUCKET
# ============================================================

df_rec["LIABS_MOST_LIKELY_BUCKET"] = df_rec[BUCKET_COLS].idxmax(axis=1)
df_rec["LIABS_MOST_LIKELY_BUCKET_PROB"] = df_rec[BUCKET_COLS].max(axis=1)


# ============================================================
# 13. BUSINESS STATUS / EXPLANATION
# ============================================================

conditions = [
    (
        (df_rec["LIABILITIES_POTENTIAL_RAW"] > 0)
        & (df_rec[PROPENSITY_COL] < 0.20)
        & (df_rec["LIABS_ZERO_BUCKET_PROB"] >= 0.50)
    ),
    (
        (df_rec["LIABILITIES_POTENTIAL_RAW"] > 0)
        & (df_rec[PROPENSITY_COL] >= 0.20)
        & (df_rec[PROPENSITY_COL] < 0.50)
    ),
    (
        (df_rec["LIABILITIES_POTENTIAL_RAW"] > 0)
        & (df_rec[PROPENSITY_COL] >= 0.50)
        & (df_rec["LIABS_EXPECTED_VOLUME"] >= 500_000)
    ),
    (
        (df_rec["LIABILITIES_POTENTIAL_RAW"] == 0)
        & (df_rec[PROPENSITY_COL] >= 0.50)
    ),
    (
        (df_rec["LIABILITIES_POTENTIAL_RAW"] == 0)
        & (df_rec[PROPENSITY_COL] < 0.20)
    )
]

choices = [
    "Potential є, але propensity/bucket не підтверджують пасиви",
    "Potential є, propensity середній — частковий дисконт",
    "Potential підтверджений propensity та bucket-моделлю",
    "Propensity є, але income-модель не бачить прибутку",
    "Низький potential і низький propensity"
]

df_rec["LIABS_RECONCILIATION_STATUS"] = np.select(
    conditions,
    choices,
    default="Нейтральний кейс"
)


# ============================================================
# 14. FINAL TOTAL POTENTIAL RECALCULATION
# ============================================================

# Якщо в тебе є загальний POTENTIAL_INCOME і ти хочеш перерахувати його,
# замінивши старий LIABILITIES_POTENTIAL на скоригований:
if "POTENTIAL_INCOME" in df_rec.columns:
    df_rec["POTENTIAL_INCOME_ADJ_LIABS_ONLY"] = (
        df_rec["POTENTIAL_INCOME"]
        - df_rec["LIABILITIES_POTENTIAL_RAW"]
        + df_rec["LIABILITIES_POTENTIAL_ADJ"]
    )


# ============================================================
# 15. USEFUL COLUMNS FOR BUSINESS REVIEW
# ============================================================

business_cols = [
    ID_COL,
    "FIRM_NAME" if "FIRM_NAME" in df_rec.columns else None,
    "FIRM_TYPE" if "FIRM_TYPE" in df_rec.columns else None,
    "POTENTIAL_INCOME" if "POTENTIAL_INCOME" in df_rec.columns else None,
    "POTENTIAL_INCOME_ADJ_LIABS_ONLY" if "POTENTIAL_INCOME_ADJ_LIABS_ONLY" in df_rec.columns else None,
    "LIABILITIES_POTENTIAL_RAW",
    "LIABILITIES_POTENTIAL_ADJ",
    PROPENSITY_COL,
    "LIABS_EXPECTED_VOLUME",
    "LIABS_BUCKET_FACTOR",
    "LIABS_ZERO_BUCKET_PROB",
    "LIABS_MOST_LIKELY_BUCKET",
    "LIABS_MOST_LIKELY_BUCKET_PROB",
    "LIABS_RECONCILIATION_STATUS"
]

business_cols = [col for col in business_cols if col is not None and col in df_rec.columns]

df_liabs_business = df_rec[business_cols].copy()


# ============================================================
# 16. BASIC DIAGNOSTICS
# ============================================================

print("Rows:", len(df_rec))
print()
print("Raw liabilities potential sum:", df_rec["LIABILITIES_POTENTIAL_RAW"].sum())
print("Adjusted liabilities potential sum:", df_rec["LIABILITIES_POTENTIAL_ADJ"].sum())
print("Difference:", df_rec["LIABILITIES_POTENTIAL_ADJ"].sum() - df_rec["LIABILITIES_POTENTIAL_RAW"].sum())
print()

print("Raw liabilities potential mean:", df_rec["LIABILITIES_POTENTIAL_RAW"].mean())
print("Adjusted liabilities potential mean:", df_rec["LIABILITIES_POTENTIAL_ADJ"].mean())
print()

print("Share with raw liabilities potential > 0:", (df_rec["LIABILITIES_POTENTIAL_RAW"] > 0).mean())
print("Share with adjusted liabilities potential > 0:", (df_rec["LIABILITIES_POTENTIAL_ADJ"] > 0).mean())
print()

print("Most likely bucket distribution:")
print(df_rec["LIABS_MOST_LIKELY_BUCKET"].value_counts(normalize=True).sort_index())
print()

print("Reconciliation status distribution:")
print(df_rec["LIABS_RECONCILIATION_STATUS"].value_counts(normalize=True))


# ============================================================
# 17. TOP CLIENTS AFTER RECONCILIATION
# ============================================================

df_liabs_top = df_liabs_business.sort_values(
    "LIABILITIES_POTENTIAL_ADJ",
    ascending=False
).head(50)

df_liabs_top.head(20)