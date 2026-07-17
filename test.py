import re
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. ШЛЯХИ
# ============================================================

LIAB_ROOT = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_Liabilities_external_clients"
)

ASSETS_ROOT = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_External_Assets"
)

FX_ROOT = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_External_FX\Results\Models"
)

INCOME_ROOT = Path(
    r"M:\Controlling\Data_Science_Projects\Income_Data"
)

DATASET_EXP_PATH = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_Liabilities\Data"
    r"\dataset_2026_06_wo_income.csv"
)

OUTPUT_PATH = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_Liabilities\Data"
    r"\validation_all_new.csv"
)


# ============================================================
# 2. ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================

def normalize_identifycode(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA
        })
        .str.zfill(8)
    )


def normalize_contragentid(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA
        })
    )


def extract_month(path):
    match = re.search(r"20\d{2}_\d{2}", str(path))

    if match:
        return match.group()

    return pd.NA


def read_table(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(path, low_memory=False)


def read_score_file(path, product, possible_score_columns):
    df = read_table(path)

    if "IDENTIFYCODE" not in df.columns:
        print(f"SKIP: немає IDENTIFYCODE — {path}")
        return None

    score_column = next(
        (
            column
            for column in possible_score_columns
            if column in df.columns
        ),
        None
    )

    if score_column is None:
        print(f"SKIP: немає score-колонки — {path}")
        return None

    selected_columns = ["IDENTIFYCODE", score_column]

    if "CONTRAGENTID" in df.columns:
        selected_columns.append("CONTRAGENTID")

    result = df[selected_columns].copy()

    result["IDENTIFYCODE"] = normalize_identifycode(
        result["IDENTIFYCODE"]
    )

    if "CONTRAGENTID" in result.columns:
        result["CONTRAGENTID"] = normalize_contragentid(
            result["CONTRAGENTID"]
        )
    else:
        result["CONTRAGENTID"] = pd.NA

    result["score"] = pd.to_numeric(
        result[score_column],
        errors="coerce"
    )

    result["product"] = product
    result["score_month"] = extract_month(path)

    result = result[
        [
            "IDENTIFYCODE",
            "CONTRAGENTID",
            "product",
            "score",
            "score_month"
        ]
    ]

    return result.dropna(
        subset=["IDENTIFYCODE", "score"]
    )


# ============================================================
# 3. ЗНАХОДИМО ВСІ ФАЙЛИ
# ============================================================

liab_files = sorted(
    LIAB_ROOT.rglob("real_combined_result.csv")
)

assets_files = sorted(
    ASSETS_ROOT.rglob("model_*.parquet")
)

fx_files = sorted(
    FX_ROOT.rglob("fx_external_*_otpay.csv")
)

income_files = sorted(
    INCOME_ROOT.glob("income_wide_corporate_clients_*.csv")
)

print("Liabilities files:", len(liab_files))
print("Assets files:", len(assets_files))
print("FX files:", len(fx_files))
print("Income files:", len(income_files))


# ============================================================
# 4. ЗЧИТУЄМО ВСІ СКОРИНГИ
# ============================================================

score_parts = []


for path in liab_files:
    try:
        part = read_score_file(
            path=path,
            product="Liabilities",
            possible_score_columns=["PRIMARY"]
        )

        if part is not None:
            score_parts.append(part)
            print(f"LIAB OK: {path.parent.name}, rows={len(part):,}")

    except Exception as error:
        print(f"LIAB ERROR: {path} -> {error}")


for path in assets_files:
    try:
        part = read_score_file(
            path=path,
            product="Assets",
            possible_score_columns=["PRIMARY"]
        )

        if part is not None:
            score_parts.append(part)
            print(f"ASSETS OK: {path.parent.name}, rows={len(part):,}")

    except Exception as error:
        print(f"ASSETS ERROR: {path} -> {error}")


for path in fx_files:
    try:
        part = read_score_file(
            path=path,
            product="FX",
            possible_score_columns=[
                "PROB_TO_FX",
                "PRIMARY",
                "FX_PRIMARY"
            ]
        )

        if part is not None:
            score_parts.append(part)
            print(f"FX OK: {path.parent.name}, rows={len(part):,}")

    except Exception as error:
        print(f"FX ERROR: {path} -> {error}")


if not score_parts:
    raise ValueError("Не вдалося зчитати жодного скорингового файлу")


scores_long = pd.concat(
    score_parts,
    ignore_index=True
)

scores_long["score_period"] = pd.to_datetime(
    scores_long["score_month"] + "_01",
    format="%Y_%m_%d",
    errors="coerce"
)


# Якщо в одному місяці клієнт зустрічається декілька разів
scores_long = (
    scores_long
    .sort_values("score_period")
    .drop_duplicates(
        subset=[
            "IDENTIFYCODE",
            "product",
            "score_month"
        ],
        keep="last"
    )
)


# Беремо останній доступний скор окремо по кожному продукту
latest_scores = (
    scores_long
    .sort_values("score_period")
    .drop_duplicates(
        subset=["IDENTIFYCODE", "product"],
        keep="last"
    )
)


scores_wide = (
    latest_scores
    .pivot(
        index="IDENTIFYCODE",
        columns="product",
        values="score"
    )
    .reset_index()
    .rename(
        columns={
            "Liabilities": "LIAB_PRIMARY",
            "Assets": "ASSETS_PRIMARY",
            "FX": "FX_PRIMARY"
        }
    )
)

scores_wide.columns.name = None


# Додаємо відсутні колонки, якщо для продукту не було файлів
for column in [
    "LIAB_PRIMARY",
    "ASSETS_PRIMARY",
    "FX_PRIMARY"
]:
    if column not in scores_wide.columns:
        scores_wide[column] = np.nan


print("\nУнікальних клієнтів у всіх скорингах:", len(scores_wide))


# ============================================================
# 5. DATASET_EXP ТА FX-ТАРГЕТ
# ============================================================

dataset_exp = pd.read_csv(
    DATASET_EXP_PATH,
    dtype={
        "IDENTIFYCODE": "string",
        "CONTRAGENTID": "string"
    },
    low_memory=False
)

dataset_exp["IDENTIFYCODE"] = normalize_identifycode(
    dataset_exp["IDENTIFYCODE"]
)

dataset_exp["CONTRAGENTID"] = normalize_contragentid(
    dataset_exp["CONTRAGENTID"]
)

dataset_exp["FX_NB_6M"] = pd.to_numeric(
    dataset_exp["FX_NB_6M"],
    errors="coerce"
).fillna(0)


fx_usage = (
    dataset_exp
    .groupby("IDENTIFYCODE", as_index=False)
    .agg(
        USED_FX=(
            "FX_NB_6M",
            lambda values: values.gt(0).any()
        )
    )
)


# ============================================================
# 6. МАПА IDENTIFYCODE — CONTRAGENTID
# ============================================================

# Мапа зі скорингових файлів
score_id_map = (
    scores_long[
        ["IDENTIFYCODE", "CONTRAGENTID"]
    ]
    .dropna()
    .drop_duplicates()
)

score_id_map["source_priority"] = 1


# Dataset_exp має вищий пріоритет
exp_id_map = (
    dataset_exp[
        ["IDENTIFYCODE", "CONTRAGENTID"]
    ]
    .dropna()
    .drop_duplicates()
)

exp_id_map["source_priority"] = 2


id_map = (
    pd.concat(
        [score_id_map, exp_id_map],
        ignore_index=True
    )
    .sort_values("source_priority")
    .drop_duplicates(
        subset="IDENTIFYCODE",
        keep="last"
    )
    .drop(columns="source_priority")
)


# ============================================================
# 7. УСІ INCOME-ФАЙЛИ
# ============================================================

income_usage_parts = []

for path in income_files:
    try:
        income = pd.read_csv(
            path,
            usecols=lambda column: column in {
                "CONTRAGENTID",
                "INCOME_LIABILITIES",
                "INCOME_ASSETS"
            },
            dtype={"CONTRAGENTID": "string"},
            low_memory=False
        )

        if "CONTRAGENTID" not in income.columns:
            print(f"INCOME SKIP: немає CONTRAGENTID — {path}")
            continue

        income["CONTRAGENTID"] = normalize_contragentid(
            income["CONTRAGENTID"]
        )

        for column in [
            "INCOME_LIABILITIES",
            "INCOME_ASSETS"
        ]:
            if column not in income.columns:
                income[column] = 0

            income[column] = pd.to_numeric(
                income[column],
                errors="coerce"
            ).fillna(0)

        month_usage = (
            income
            .groupby("CONTRAGENTID", as_index=False)
            .agg(
                USED_LIABILITIES=(
                    "INCOME_LIABILITIES",
                    lambda values: values.gt(0).any()
                ),
                USED_ASSETS=(
                    "INCOME_ASSETS",
                    lambda values: values.gt(0).any()
                )
            )
        )

        income_usage_parts.append(month_usage)

        print(
            f"INCOME OK: {extract_month(path)}, "
            f"clients={len(month_usage):,}"
        )

    except Exception as error:
        print(f"INCOME ERROR: {path} -> {error}")


if income_usage_parts:
    income_usage = (
        pd.concat(
            income_usage_parts,
            ignore_index=True
        )
        .groupby("CONTRAGENTID", as_index=False)
        .agg(
            USED_LIABILITIES=("USED_LIABILITIES", "max"),
            USED_ASSETS=("USED_ASSETS", "max")
        )
    )

else:
    income_usage = pd.DataFrame(
        columns=[
            "CONTRAGENTID",
            "USED_LIABILITIES",
            "USED_ASSETS"
        ]
    )


# ============================================================
# 8. ФІНАЛЬНИЙ ДАТАСЕТ
# ============================================================

# Базою є ВСІ клієнти з усіх скорингових файлів
validation_all = (
    scores_wide
    .merge(
        id_map,
        how="left",
        on="IDENTIFYCODE"
    )
    .merge(
        income_usage,
        how="left",
        on="CONTRAGENTID"
    )
    .merge(
        fx_usage,
        how="left",
        on="IDENTIFYCODE"
    )
)


target_columns = [
    "USED_LIABILITIES",
    "USED_ASSETS",
    "USED_FX"
]

validation_all[target_columns] = (
    validation_all[target_columns]
    .fillna(False)
    .astype(bool)
)


PRODUCT_FLAGS = {
    "Liabilities": "USED_LIABILITIES",
    "Assets": "USED_ASSETS",
    "FX": "USED_FX"
}


def get_actual_product(row):
    products = [
        product
        for product, flag_column in PRODUCT_FLAGS.items()
        if row[flag_column]
    ]

    return ", ".join(products) if products else "None"


validation_all["actual_product"] = validation_all.apply(
    get_actual_product,
    axis=1
)

validation_all["n_actual_products"] = (
    validation_all[target_columns]
    .sum(axis=1)
)


validation_all = validation_all[
    [
        "IDENTIFYCODE",
        "CONTRAGENTID",
        "LIAB_PRIMARY",
        "ASSETS_PRIMARY",
        "FX_PRIMARY",
        "USED_LIABILITIES",
        "USED_ASSETS",
        "USED_FX",
        "actual_product",
        "n_actual_products"
    ]
].sort_values(
    ["CONTRAGENTID", "IDENTIFYCODE"],
    na_position="last"
).reset_index(drop=True)


# ============================================================
# 9. ПЕРЕВІРКИ
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Кількість рядків:", len(validation_all))
print(
    "Унікальних IDENTIFYCODE:",
    validation_all["IDENTIFYCODE"].nunique()
)

print(
    "Без CONTRAGENTID:",
    validation_all["CONTRAGENTID"].isna().sum()
)

print("\nНаявність скорів:")
print(
    validation_all[
        [
            "LIAB_PRIMARY",
            "ASSETS_PRIMARY",
            "FX_PRIMARY"
        ]
    ]
    .notna()
    .sum()
)

print("\nРозподіл actual_product:")
print(
    validation_all["actual_product"]
    .value_counts(dropna=False)
)


# Перевірка, що merge не видалив клієнтів
assert len(validation_all) == scores_wide["IDENTIFYCODE"].nunique()


# ============================================================
# 10. ЗБЕРЕЖЕННЯ
# ============================================================

validation_all.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nЗбережено: {OUTPUT_PATH}")