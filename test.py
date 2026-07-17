from pathlib import Path
import pandas as pd


# ============================================================
# НАЛАШТУВАННЯ
# ============================================================

START_MONTH = "2025_05"
VALID_MONTH = "2026_06"

# На скріншоті для 2026_06 Assets лежить у папці 2026_05.
# Якщо надалі Assets-файл відповідає поточному місяцю, постав 0.
ASSETS_LAG_MONTHS = 1

LIABS_BASE = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_liabilities_external_clients"
)

ASSETS_BASE = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_External_Assets"
)

FX_BASE = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_External_FX\Results\Models"
)

INCOME_BASE = Path(
    r"M:\Controlling\Data_Science_Projects\Income_Data"
)

EXP_BASE = Path(
    r"M:\Controlling\Data_Science_Projects\Corp_liabilities\Data"
)


# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================

def shift_month(month, shift):
    period = pd.Period(
        month.replace("_", "-"),
        freq="M"
    )

    return (period + shift).strftime("%Y_%m")


def normalize_identifycode(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def normalize_contragentid(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


# ============================================================
# ЗЧИТУВАННЯ INCOME
# ============================================================

def load_income(month, suffix):
    path = (
        INCOME_BASE
        / f"income_wide_corporate_clients_{month}.csv"
    )

    income = pd.read_csv(
        path,
        dtype={"CONTRAGENTID": "string"},
        usecols=[
            "CONTRAGENTID",
            "INCOME_LIABILITIES",
            "INCOME_ASSETS",
            "COM_CORP_FX_FOR_PAY",
        ],
    )

    income["CONTRAGENTID"] = normalize_contragentid(
        income["CONTRAGENTID"]
    )

    income_columns = [
        "INCOME_LIABILITIES",
        "INCOME_ASSETS",
        "COM_CORP_FX_FOR_PAY",
    ]

    income[income_columns] = income[income_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)

    # Якщо на одного клієнта декілька рядків
    income = (
        income
        .groupby("CONTRAGENTID", as_index=False)[income_columns]
        .sum()
    )

    income = income.rename(
        columns={
            column: f"{column}_{suffix}"
            for column in income_columns
        }
    )

    return income


# ============================================================
# ЗЧИТУВАННЯ DATASET_EXP
# ============================================================

def load_dataset_exp(month, suffix):
    path = (
        EXP_BASE
        / f"dataset_{month}_wo_income.csv"
    )

    exp = pd.read_csv(
        path,
        dtype={
            "CONTRAGENTID": "string",
            "IDENTIFYCODE": "string",
        },
        usecols=[
            "CONTRAGENTID",
            "IDENTIFYCODE",
            "FX_NB_6M",
        ],
    )

    exp["CONTRAGENTID"] = normalize_contragentid(
        exp["CONTRAGENTID"]
    )

    exp["IDENTIFYCODE"] = normalize_identifycode(
        exp["IDENTIFYCODE"]
    )

    exp["FX_NB_6M"] = pd.to_numeric(
        exp["FX_NB_6M"],
        errors="coerce",
    ).fillna(0)

    # Один запис на CONTRAGENTID
    exp = (
        exp
        .sort_values("FX_NB_6M")
        .drop_duplicates(
            subset="CONTRAGENTID",
            keep="last",
        )
    )

    exp = exp.rename(
        columns={
            "FX_NB_6M": f"FX_NB_6M_{suffix}"
        }
    )

    return exp


# ============================================================
# TARGET ДЛЯ ОДНОГО МІСЯЦЯ
# ============================================================

def build_month_target(month):
    previous_month = shift_month(month, -1)

    income_current = load_income(
        month=month,
        suffix="CURRENT",
    )

    income_previous = load_income(
        month=previous_month,
        suffix="PREVIOUS",
    )

    exp_current = load_dataset_exp(
        month=month,
        suffix="CURRENT",
    )

    exp_previous = load_dataset_exp(
        month=previous_month,
        suffix="PREVIOUS",
    )

    # Основою є клієнти з dataset_exp поточного місяця
    data = (
        exp_current
        .merge(
            income_current,
            on="CONTRAGENTID",
            how="left",
        )
        .merge(
            income_previous,
            on="CONTRAGENTID",
            how="left",
        )
        .merge(
            exp_previous[
                [
                    "CONTRAGENTID",
                    "FX_NB_6M_PREVIOUS",
                ]
            ],
            on="CONTRAGENTID",
            how="left",
        )
    )

    numeric_columns = [
        "INCOME_LIABILITIES_CURRENT",
        "INCOME_ASSETS_CURRENT",
        "COM_CORP_FX_FOR_PAY_CURRENT",
        "FX_NB_6M_CURRENT",
        "INCOME_LIABILITIES_PREVIOUS",
        "INCOME_ASSETS_PREVIOUS",
        "COM_CORP_FX_FOR_PAY_PREVIOUS",
        "FX_NB_6M_PREVIOUS",
    ]

    data[numeric_columns] = data[numeric_columns].fillna(0)

    # Активність у поточному місяці
    data["LIAB_CURRENT"] = (
        data["INCOME_LIABILITIES_CURRENT"] > 0
    )

    data["ASSETS_CURRENT"] = (
        data["INCOME_ASSETS_CURRENT"] > 0
    )

    data["FX_CURRENT"] = (
        (data["COM_CORP_FX_FOR_PAY_CURRENT"] > 0)
        | (data["FX_NB_6M_CURRENT"] > 0)
    )

    # Активність у попередньому місяці
    data["LIAB_PREVIOUS"] = (
        data["INCOME_LIABILITIES_PREVIOUS"] > 0
    )

    data["ASSETS_PREVIOUS"] = (
        data["INCOME_ASSETS_PREVIOUS"] > 0
    )

    data["FX_PREVIOUS"] = (
        (data["COM_CORP_FX_FOR_PAY_PREVIOUS"] > 0)
        | (data["FX_NB_6M_PREVIOUS"] > 0)
    )

    # Новий продукт з'явився саме в поточному місяці
    data["NEW_LIABILITIES"] = (
        data["LIAB_CURRENT"]
        & ~data["LIAB_PREVIOUS"]
    )

    data["NEW_ASSETS"] = (
        data["ASSETS_CURRENT"]
        & ~data["ASSETS_PREVIOUS"]
    )

    data["NEW_FX"] = (
        data["FX_CURRENT"]
        & ~data["FX_PREVIOUS"]
    )

    # Чи був клієнт активним до поточного місяця
    data["WAS_ACTIVE_BEFORE"] = (
        data["LIAB_PREVIOUS"]
        | data["ASSETS_PREVIOUS"]
        | data["FX_PREVIOUS"]
    )

    # Один IDENTIFYCODE може зустрічатися декілька разів
    data = (
        data
        .groupby("IDENTIFYCODE", as_index=False)
        .agg(
            CONTRAGENTID=("CONTRAGENTID", "first"),
            NEW_LIABILITIES=("NEW_LIABILITIES", "max"),
            NEW_ASSETS=("NEW_ASSETS", "max"),
            NEW_FX=("NEW_FX", "max"),
            WAS_ACTIVE_BEFORE=("WAS_ACTIVE_BEFORE", "max"),
        )
    )

    def get_actual_product(row):
        products = []

        if row["NEW_LIABILITIES"]:
            products.append("LIABILITIES")

        if row["NEW_ASSETS"]:
            products.append("ASSETS")

        if row["NEW_FX"]:
            products.append("FX")

        return ", ".join(products) if products else "NONE"

    data["ACTUAL_PRODUCT"] = data.apply(
        get_actual_product,
        axis=1,
    )

    return data


# ============================================================
# ЗЧИТУВАННЯ PROPENSITY-МОДЕЛЕЙ
# ============================================================

def load_month_scores(month):
    assets_month = shift_month(
        month,
        -ASSETS_LAG_MONTHS,
    )

    liabs_path = (
        LIABS_BASE
        / month
        / "real_combined_result.csv"
    )

    assets_path = (
        ASSETS_BASE
        / assets_month
        / f"model_{assets_month}.parquet"
    )

    fx_path = (
        FX_BASE
        / month
        / f"fx_external_{month}.parquet"
    )

    paths = [
        liabs_path,
        assets_path,
        fx_path,
    ]

    missing_paths = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing_paths:
        print(f"{month}: немає файлів")

        for path in missing_paths:
            print(path)

        return None

    liabs = pd.read_csv(
        liabs_path,
        dtype={"IDENTIFYCODE": "string"},
        usecols=["IDENTIFYCODE", "PRIMARY"],
    ).rename(
        columns={"PRIMARY": "LIAB_PRIMARY"}
    )

    assets = pd.read_parquet(
        assets_path,
        columns=["IDENTIFYCODE", "PRIMARY"],
    ).rename(
        columns={"PRIMARY": "ASSETS_PRIMARY"}
    )

    fx = pd.read_parquet(
        fx_path,
        columns=["IDENTIFYCODE", "PROB_TO_FX"],
    ).rename(
        columns={"PROB_TO_FX": "FX_PRIMARY"}
    )

    score_frames = [liabs, assets, fx]

    for frame in score_frames:
        frame["IDENTIFYCODE"] = normalize_identifycode(
            frame["IDENTIFYCODE"]
        )

        frame.drop_duplicates(
            subset="IDENTIFYCODE",
            keep="last",
            inplace=True,
        )

    scores = (
        liabs
        .merge(
            assets,
            on="IDENTIFYCODE",
            how="outer",
        )
        .merge(
            fx,
            on="IDENTIFYCODE",
            how="outer",
        )
    )

    scores = scores.dropna(
        subset=[
            "LIAB_PRIMARY",
            "ASSETS_PRIMARY",
            "FX_PRIMARY",
        ],
        how="all",
    )

    return scores


# ============================================================
# ФОРМУВАННЯ ВСІЄЇ ВИБІРКИ
# ============================================================

months = (
    pd.period_range(
        START_MONTH.replace("_", "-"),
        VALID_MONTH.replace("_", "-"),
        freq="M",
    )
    .strftime("%Y_%m")
    .tolist()
)

monthly_datasets = []

for month in months:
    try:
        scores = load_month_scores(month)

        if scores is None:
            continue

        target = build_month_target(month)

        month_data = scores.merge(
            target,
            on="IDENTIFYCODE",
            how="inner",
        )

        # Модель розрахована лише для зовнішніх клієнтів.
        # Тому прибираємо тих, хто вже мав продукт раніше.
        month_data = month_data[
            ~month_data["WAS_ACTIVE_BEFORE"]
        ].copy()

        month_data.insert(
            0,
            "SCORE_MONTH",
            month,
        )

        monthly_datasets.append(month_data)

        print(
            f"{month}: "
            f"{len(month_data)} клієнтів, "
            f"{(month_data['ACTUAL_PRODUCT'] != 'NONE').sum()} залучених"
        )

    except FileNotFoundError as error:
        print(f"{month}: пропущено")
        print(error)


full_dataset = pd.concat(
    monthly_datasets,
    ignore_index=True,
)

full_dataset["IDENTIFYCODE"] = (
    full_dataset["IDENTIFYCODE"].astype("category")
)


# ============================================================
# TRAIN / VALIDATION
# ============================================================

train_data = full_dataset[
    full_dataset["SCORE_MONTH"] < VALID_MONTH
].reset_index(drop=True)

validation_data = full_dataset[
    full_dataset["SCORE_MONTH"] == VALID_MONTH
].reset_index(drop=True)


print("\nTrain:", train_data.shape)
print("Validation:", validation_data.shape)

print("\nTrain targets:")
print(
    train_data["ACTUAL_PRODUCT"]
    .value_counts(dropna=False)
)

print("\nValidation targets:")
print(
    validation_data["ACTUAL_PRODUCT"]
    .value_counts(dropna=False)
)