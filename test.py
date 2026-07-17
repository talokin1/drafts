from pathlib import Path
import pandas as pd


# =========================================================
# НАЛАШТУВАННЯ
# =========================================================

START_MONTH = "2025_05"
VALID_MONTH = "2026_06"

LIABS_BASE = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_liabilities_external_clients"
)

ASSETS_BASE = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_External_Assets"
)

FX_BASE = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_External_FX\Results\Models"
)

# Префікси колонок у income_data
INCOME_PREFIXES = {
    "LIABILITIES": "INC_LIABS_",
    "ASSETS": "INC_ASSETS_",
    "FX": "INCOME_",       # заміни, якщо FX-дохід має інший префікс
}

# Якщо dataset_exp має окрему колонку місяця, наприклад OBS_MONTH
EXP_MONTH_COL = None
# EXP_MONTH_COL = "OBS_MONTH"


# =========================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# =========================================================

def normalize_id(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def get_model_paths(month):
    return {
        "liabs": (
            LIABS_BASE
            / month
            / "real_combined_result.csv"
        ),
        "assets": (
            ASSETS_BASE
            / month
            / f"model_{month}.parquet"
        ),
        "fx": (
            FX_BASE
            / month
            / f"fx_external_{month}.parquet"
        ),
    }


def load_month_scores(month):
    paths = get_model_paths(month)

    missing = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing:
        print(f"{month}: пропущено — немає файлів:")
        for path in missing:
            print(path)
        return None

    liabs = pd.read_csv(
        paths["liabs"],
        dtype={"IDENTIFYCODE": "string"},
    )[["IDENTIFYCODE", "PRIMARY"]].rename(
        columns={"PRIMARY": "LIAB_PRIMARY"}
    )

    assets = pd.read_parquet(
        paths["assets"],
        columns=["IDENTIFYCODE", "PRIMARY"],
    ).rename(
        columns={"PRIMARY": "ASSETS_PRIMARY"}
    )

    fx = pd.read_parquet(
        paths["fx"],
        columns=["IDENTIFYCODE", "PROB_TO_FX"],
    ).rename(
        columns={"PROB_TO_FX": "FX_PRIMARY"}
    )

    frames = [liabs, assets, fx]

    for frame in frames:
        frame["IDENTIFYCODE"] = normalize_id(
            frame["IDENTIFYCODE"]
        )
        frame.drop_duplicates(
            subset="IDENTIFYCODE",
            keep="last",
            inplace=True,
        )

    scores = (
        liabs
        .merge(assets, on="IDENTIFYCODE", how="outer")
        .merge(fx, on="IDENTIFYCODE", how="outer")
    )

    return scores


def started_product_in_month(data, prefix, month):
    current_col = f"{prefix}{month}"

    if current_col not in data.columns:
        return pd.Series(False, index=data.index)

    previous_cols = [
        col
        for col in data.columns
        if (
            col.startswith(prefix)
            and len(col) == len(prefix) + 7
            and col[-7:] < month
        )
    ]

    current_income = data[current_col].fillna(0).ne(0)

    if previous_cols:
        had_income_before = (
            data[previous_cols]
            .fillna(0)
            .ne(0)
            .any(axis=1)
        )
    else:
        had_income_before = pd.Series(
            False,
            index=data.index,
        )

    return current_income & ~had_income_before


def get_fx_clients(month):
    exp_month = dataset_exp

    if EXP_MONTH_COL is not None:
        exp_month = dataset_exp[
            dataset_exp[EXP_MONTH_COL]
            .astype("string")
            .str.replace("-", "_")
            .eq(month)
        ]

    return set(
        exp_month.loc[
            pd.to_numeric(
                exp_month["FX_NB_6M"],
                errors="coerce",
            ).fillna(0).ne(0),
            "IDENTIFYCODE",
        ]
    )


def build_actual_products(month):
    masks = {
        product: started_product_in_month(
            income_data,
            prefix,
            month,
        )
        for product, prefix in INCOME_PREFIXES.items()
    }

    # Для FX недостатньо лише доходу:
    # клієнт також має бути в dataset_exp з FX_NB_6M != 0
    fx_clients = get_fx_clients(month)

    masks["FX"] = (
        masks["FX"]
        & income_data["IDENTIFYCODE"].isin(fx_clients)
    )

    actual_product = pd.Series(
        pd.NA,
        index=income_data.index,
        dtype="string",
    )

    for product, mask in masks.items():
        add_to_existing = mask & actual_product.notna()
        add_first = mask & actual_product.isna()

        actual_product.loc[add_to_existing] = (
            actual_product.loc[add_to_existing]
            + ", "
            + product
        )

        actual_product.loc[add_first] = product

    result = income_data.loc[
        actual_product.notna(),
        ["IDENTIFYCODE"],
    ].copy()

    result["ACTUAL_PRODUCT"] = actual_product[
        actual_product.notna()
    ]

    return result


# =========================================================
# ПІДГОТОВКА INCOME_DATA ТА DATASET_EXP
# =========================================================

income_data = income_data.copy()
dataset_exp = dataset_exp.copy()

income_data["IDENTIFYCODE"] = normalize_id(
    income_data["IDENTIFYCODE"]
)

dataset_exp["IDENTIFYCODE"] = normalize_id(
    dataset_exp["IDENTIFYCODE"]
)

# Знаходимо всі помісячні колонки доходу
income_month_cols = [
    col
    for col in income_data.columns
    if any(
        col.startswith(prefix)
        and len(col) == len(prefix) + 7
        for prefix in INCOME_PREFIXES.values()
    )
]

income_data[income_month_cols] = income_data[
    income_month_cols
].apply(
    pd.to_numeric,
    errors="coerce",
)

# Один рядок на клієнта
income_data = (
    income_data[
        ["IDENTIFYCODE"] + income_month_cols
    ]
    .groupby("IDENTIFYCODE", as_index=False)
    .sum(min_count=1)
)


# =========================================================
# ФОРМУВАННЯ ПОМІСЯЧНОЇ ВИБІРКИ
# =========================================================

months = (
    pd.period_range(
        START_MONTH.replace("_", "-"),
        VALID_MONTH.replace("_", "-"),
        freq="M",
    )
    .strftime("%Y_%m")
    .tolist()
)

monthly_samples = []

for month in months:
    scores = load_month_scores(month)

    if scores is None:
        continue

    actual = build_actual_products(month)

    month_sample = scores.merge(
        actual,
        on="IDENTIFYCODE",
        how="inner",
    )

    score_cols = [
        "LIAB_PRIMARY",
        "ASSETS_PRIMARY",
        "FX_PRIMARY",
    ]

    # Залишаємо лише клієнтів, для яких у цьому
    # конкретному місяці був хоча б один скор
    month_sample = month_sample[
        month_sample[score_cols].notna().any(axis=1)
    ].copy()

    month_sample.insert(1, "SCORE_MONTH", month)

    monthly_samples.append(month_sample)

    print(
        month,
        "clients:",
        len(month_sample),
    )


full_sample = pd.concat(
    monthly_samples,
    ignore_index=True,
)

full_sample["IDENTIFYCODE"] = (
    full_sample["IDENTIFYCODE"].astype("category")
)

train_data = full_sample[
    full_sample["SCORE_MONTH"] < VALID_MONTH
].reset_index(drop=True)

validation_data = full_sample[
    full_sample["SCORE_MONTH"] == VALID_MONTH
].reset_index(drop=True)


print("Train:", train_data.shape)
print("Validation:", validation_data.shape)

display(train_data.head())
display(validation_data.head())