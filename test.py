import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# 1. КОНФІГУРАЦІЯ
# ============================================================

LIABS_FEATURES = [
    "CASH_CUR",
    "CASH_PREV",
    "CASH_DIF",
    "ALR%_CUR",
    "ALR%_PREV",
    "ALR%_DIF",
    "ASSETS_PREV",
    "ASSETS_DIF",
    "ROE%_CUR",
    "ROE%_DIF",
    "TAT_CUR",
    "NB_EMP",
    "CDR_PREV"
]

ASSETS_FEATURES = [
    "CURRENT_ASSETS_CUR",
    "ASSETS_CUR",
    "INVENTORY_CUR",
    "PAYABLES_DIF",
    "DSO_PREV",
    "CR%_DIF",
    "LIQUID_ASSETS_DIF",
    "OPM%_CUR",
    "OPM%_DIF",
    "FIXED_ASSETS_PREV",
    "FIXED_ASSETS_DIF",
    "LTFR_DIF"
]

FX_FEATURES = [
    "IMPORT_USD",
    "EXPORT_USD",
    "A1165",
    "A2120",
    "A2160",
    "A2285",
    "DIO_CUR",
    "DIO_PREV",
    "TAT_CUR",
    "B2120"
]


PRODUCT_CONFIG = {
    "Liabilities": {
        "propensity_col": "LIAB_PRIMARY",
        "target_col": "NEW_LIABILITIES",
        "financial_col": "LIAB_FIN_SCORE",
        "final_col": "LIAB_FINAL_SCORE",
        "recommendation_col": "REC_LIABILITIES",
        "features": LIABS_FEATURES,
        "propensity_weight": 0.70,
        "financial_weight": 0.30,
        "propensity_floor": 0.20
    },

    "Assets": {
        "propensity_col": "ASSETS_PRIMARY",
        "target_col": "NEW_ASSETS",
        "financial_col": "ASSETS_FIN_SCORE",
        "final_col": "ASSETS_FINAL_SCORE",
        "recommendation_col": "REC_ASSETS",
        "features": ASSETS_FEATURES,
        "propensity_weight": 0.65,
        "financial_weight": 0.35,
        "propensity_floor": 0.20
    },

    "FX": {
        "propensity_col": "FX_PRIMARY",
        "target_col": "NEW_FX",
        "financial_col": "FX_FIN_SCORE",
        "final_col": "FX_FINAL_SCORE",
        "recommendation_col": "REC_FX",
        "features": FX_FEATURES,
        "propensity_weight": 0.80,
        "financial_weight": 0.20,
        "propensity_floor": 0.40
    }
}

ALL_FINANCIAL_FEATURES = list(dict.fromkeys(
    LIABS_FEATURES
    + ASSETS_FEATURES
    + FX_FEATURES
))


# ============================================================
# 2. ПІДГОТОВКА ФІНАНСОВИХ ДАНИХ
# ============================================================

def normalize_identifycode(df, col="IDENTIFYCODE"):
    df = df.copy()

    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )

    return df


def prepare_financial_data(fin_ind, fin_ind_wo):
    financial = pd.concat(
        [fin_ind.copy(), fin_ind_wo.copy()],
        ignore_index=True,
        sort=False
    )

    financial = normalize_identifycode(financial)

    for col in ALL_FINANCIAL_FEATURES:
        if col not in financial.columns:
            financial[col] = np.nan

        financial[col] = pd.to_numeric(
            financial[col],
            errors="coerce"
        )

    # Вибираємо найповніший запис клієнта
    financial["_FEATURES_FILLED"] = (
        financial[ALL_FINANCIAL_FEATURES]
        .notna()
        .sum(axis=1)
    )

    if "REP_YEAR" in financial.columns:
        financial["_REP_YEAR_SORT"] = pd.to_numeric(
            financial["REP_YEAR"],
            errors="coerce"
        )
    else:
        financial["_REP_YEAR_SORT"] = np.nan

    if "DFILL" in financial.columns:
        financial["_DFILL_SORT"] = pd.to_datetime(
            financial["DFILL"],
            errors="coerce"
        )
    else:
        financial["_DFILL_SORT"] = pd.NaT

    financial = (
        financial
        .sort_values(
            [
                "IDENTIFYCODE",
                "_FEATURES_FILLED",
                "_REP_YEAR_SORT",
                "_DFILL_SORT"
            ],
            ascending=[True, False, False, False],
            na_position="last"
        )
        .drop_duplicates(
            subset="IDENTIFYCODE",
            keep="first"
        )
        .reset_index(drop=True)
    )

    return financial


financial_base = prepare_financial_data(
    fin_ind=fin_ind,
    fin_ind_wo=fin_ind_wo
)














# ============================================================
# 3. WINSORIZE + PERCENTILE SCORE
# ============================================================

def fit_financial_reference(financial_df):
    reference = {}

    for feature in ALL_FINANCIAL_FEATURES:
        values = pd.to_numeric(
            financial_df[feature],
            errors="coerce"
        ).dropna()

        if values.empty:
            reference[feature] = None
            continue

        lower = values.quantile(0.01)
        upper = values.quantile(0.99)

        clipped = np.sort(
            values.clip(lower, upper).to_numpy()
        )

        reference[feature] = {
            "lower": lower,
            "upper": upper,
            "values": clipped
        }

    return reference


def calculate_financial_scores(
    financial_df,
    reference
):
    result = financial_df[
        ["IDENTIFYCODE"]
    ].copy()

    percentile_columns = {}

    for feature in ALL_FINANCIAL_FEATURES:
        output_col = f"{feature}__PCT"
        percentile_columns[feature] = output_col

        values = pd.to_numeric(
            financial_df[feature],
            errors="coerce"
        )

        result[output_col] = np.nan

        feature_reference = reference.get(feature)

        if feature_reference is None:
            continue

        valid_mask = values.notna()

        lower = feature_reference["lower"]
        upper = feature_reference["upper"]
        sorted_reference = feature_reference["values"]

        if lower == upper:
            result.loc[valid_mask, output_col] = 0.5
            continue

        clipped = values.loc[valid_mask].clip(
            lower,
            upper
        )

        result.loc[valid_mask, output_col] = (
            np.searchsorted(
                sorted_reference,
                clipped,
                side="right"
            )
            / len(sorted_reference)
        )

    # Пропуски не перетворюємо на нулі.
    # Скор — середнє лише доступних індикаторів.
    for product, config in PRODUCT_CONFIG.items():
        feature_score_cols = [
            percentile_columns[feature]
            for feature in config["features"]
        ]

        available_count = (
            result[feature_score_cols]
            .notna()
            .sum(axis=1)
        )

        result[config["financial_col"]] = (
            result[feature_score_cols]
            .mean(axis=1, skipna=True)
        )

        result[f"{product.upper()}_FIN_COVERAGE"] = (
            available_count
            / len(feature_score_cols)
        )

        result.loc[
            available_count == 0,
            config["financial_col"]
        ] = np.nan

    return result


financial_reference = fit_financial_reference(
    financial_base
)

financial_scores = calculate_financial_scores(
    financial_df=financial_base,
    reference=financial_reference
)










# ============================================================
# 4. FINAL SCORE
# ============================================================

def add_recommendation_scores(
    scores_df,
    financial_scores
):
    df = normalize_identifycode(scores_df)

    df = df.merge(
        financial_scores,
        how="left",
        on="IDENTIFYCODE",
        validate="many_to_one"
    )

    for product, config in PRODUCT_CONFIG.items():
        propensity_col = config["propensity_col"]
        financial_col = config["financial_col"]
        final_col = config["final_col"]

        df[propensity_col] = pd.to_numeric(
            df[propensity_col],
            errors="coerce"
        )

        if df[propensity_col].dropna().gt(1).any():
            raise ValueError(
                f"{propensity_col} містить значення більше 1"
            )

        propensity = df[propensity_col].clip(0, 1)
        financial = df[financial_col].clip(0, 1)

        prop_weight = config["propensity_weight"]
        fin_weight = config["financial_weight"]

        # Якщо фінансових даних немає,
        # фінальний скор дорівнює propensity
        df[final_col] = np.where(
            financial.notna(),
            prop_weight * propensity
            + fin_weight * financial,
            propensity
        )

    return df

# ============================================================
# 5. ПІДБІР FINAL THRESHOLD
# ============================================================

def tune_product_thresholds(validation_df):
    rules = {}

    threshold_grid = np.arange(
        0.10,
        0.91,
        0.01
    )

    for product, config in PRODUCT_CONFIG.items():
        propensity_col = config["propensity_col"]
        final_col = config["final_col"]
        target_col = config["target_col"]
        propensity_floor = config["propensity_floor"]

        work = validation_df[
            [propensity_col, final_col, target_col]
        ].dropna(
            subset=[propensity_col, final_col]
        )

        y_true = (
            work[target_col]
            .fillna(False)
            .astype(bool)
        )

        if y_true.sum() == 0:
            raise ValueError(
                f"Для {product} немає позитивних клієнтів"
            )

        best_result = None

        for threshold in threshold_grid:
            y_pred = (
                (work[propensity_col] >= propensity_floor)
                & (work[final_col] >= threshold)
            )

            result = {
                "threshold": float(threshold),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            }

            if (
                best_result is None
                or (
                    result["f1"],
                    result["precision"],
                    result["recall"]
                )
                > (
                    best_result["f1"],
                    best_result["precision"],
                    best_result["recall"]
                )
            ):
                best_result = result

        best_result["propensity_floor"] = propensity_floor
        rules[product] = best_result

    return rules

# ============================================================
# 6. ФІНАЛЬНЕ РІШЕННЯ
# ============================================================

def make_recommendations(
    scored_df,
    rules
):
    df = scored_df.copy()

    for product, config in PRODUCT_CONFIG.items():
        rule = rules[product]

        df[config["recommendation_col"]] = (
            df[config["propensity_col"]]
            .ge(rule["propensity_floor"])
            &
            df[config["final_col"]]
            .ge(rule["threshold"])
        )

    recommendation_mapping = {
        "Liabilities": "REC_LIABILITIES",
        "Assets": "REC_ASSETS",
        "FX": "REC_FX"
    }

    df["recommended_product"] = df.apply(
        lambda row: ", ".join(
            product
            for product, recommendation_col
            in recommendation_mapping.items()
            if row[recommendation_col]
        ) or "None",
        axis=1
    )

    df["n_recommended_products"] = df[
        list(recommendation_mapping.values())
    ].sum(axis=1)

    return df


validation_scored = add_recommendation_scores(
    scores_df=validation_all_full,
    financial_scores=financial_scores
)

# Часове розділення
months = sorted(
    validation_scored["score_month"]
    .dropna()
    .unique()
)

if len(months) < 2:
    raise ValueError(
        "Для часового train/test split потрібно мінімум 2 місяці"
    )

split_index = max(
    1,
    int(len(months) * 0.8)
)

if split_index == len(months):
    split_index = len(months) - 1

tune_months = months[:split_index]
test_months = months[split_index:]

validation_tune = validation_scored[
    validation_scored["score_month"].isin(tune_months)
].copy()

validation_test = validation_scored[
    validation_scored["score_month"].isin(test_months)
].copy()

rules = tune_product_thresholds(
    validation_tune
)

rules
















validation_predictions = make_recommendations(
    validation_test,
    rules
)

validation_predictions[
    [
        "IDENTIFYCODE",
        "actual_product",
        "recommended_product",
        "LIAB_PRIMARY",
        "LIAB_FIN_SCORE",
        "LIAB_FINAL_SCORE",
        "ASSETS_PRIMARY",
        "ASSETS_FIN_SCORE",
        "ASSETS_FINAL_SCORE",
        "FX_PRIMARY",
        "FX_FIN_SCORE",
        "FX_FINAL_SCORE"
    ]
].head()



metrics = []

for product, config in PRODUCT_CONFIG.items():
    y_true = validation_predictions[
        config["target_col"]
    ].astype(bool)

    y_pred = validation_predictions[
        config["recommendation_col"]
    ].astype(bool)

    metrics.append({
        "product": product,
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "recommended": int(y_pred.sum()),
        "actual": int(y_true.sum())
    })

metrics = pd.DataFrame(metrics)

none_true = (
    validation_predictions["n_actual_products"] == 0
)

none_pred = (
    validation_predictions["n_recommended_products"] == 0
)

none_metrics = {
    "precision": precision_score(
        none_true, none_pred, zero_division=0
    ),
    "recall": recall_score(
        none_true, none_pred, zero_division=0
    ),
    "f1": f1_score(
        none_true, none_pred, zero_division=0
    )
}

metrics, none_metrics

















CURRENT_MONTH = "2026_07"

scores_current = read_scores(CURRENT_MONTH)

current_scored = add_recommendation_scores(
    scores_df=scores_current,
    financial_scores=financial_scores
)

recommendations = make_recommendations(
    current_scored,
    rules
)

recommendations = recommendations[
    [
        "IDENTIFYCODE",
        "CONTRAGENTID",
        "recommended_product",
        "n_recommended_products",

        "LIAB_PRIMARY",
        "LIAB_FIN_SCORE",
        "LIAB_FINAL_SCORE",
        "REC_LIABILITIES",

        "ASSETS_PRIMARY",
        "ASSETS_FIN_SCORE",
        "ASSETS_FINAL_SCORE",
        "REC_ASSETS",

        "FX_PRIMARY",
        "FX_FIN_SCORE",
        "FX_FINAL_SCORE",
        "REC_FX"
    ]
]

recommendations.head()