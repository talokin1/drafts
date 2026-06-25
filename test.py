import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

product_config = {
    "Liabilities": {
        "score_col": "LIAB_PRIMARY",
        "target_col": "NEW_LIABILITIES",
        "calibrated_col": "calibrated_liabs"
    },
    
    "Assets": {
        "score_col": "ASSETS_PRIMARY",
        "target_col": "NEW_ASSETS",
        "calibrated_col": "calibrated_assets"
    },
    
    "FX": {
        "score_col": "FX_PRIMARY",
        "target_col": "NEW_FX",
        "calibrated_col": "calibrated_fx"
    }
}



train_months = [
    "2025_05",
    "2025_06",
    "2025_07",
    "2025_08",
    "2025_09",
    "2025_10",
    "2025_11",
    "2025_12",
    "2026_01"
]

test_months = [
    "2026_02",
    "2026_03",
    "2026_04"
]

train_data = validation_all[
    validation_all["score_month"].isin(train_months)
].copy()

test_data = validation_all[
    validation_all["score_month"].isin(test_months)
].copy()
print("Train rows:", len(train_data))
print("Test rows:", len(test_data))

positive_check = []

for product, cfg in product_config.items():
    
    tmp = train_data[
        train_data[cfg["score_col"]].notna()
    ].copy()
    
    positive_check.append({
        "product": product,
        "scored_clients": len(tmp),
        "positive_cases": tmp[cfg["target_col"]].sum(),
        "conversion_rate": tmp[cfg["target_col"]].mean()
    })

positive_check = pd.DataFrame(positive_check)

positive_check












calibrators = {}

MIN_POSITIVES = 15

for product, cfg in product_config.items():
    
    score_col = cfg["score_col"]
    target_col = cfg["target_col"]
    
    product_train = train_data[
        train_data[score_col].notna()
    ].copy()
    
    X = product_train[[score_col]]
    y = product_train[target_col].astype(int)
    
    positives = y.sum()
    
    if y.nunique() < 2 or positives < MIN_POSITIVES:
        print(
            f"{product}: calibration skipped "
            f"(positive cases = {positives})"
        )
        continue
    
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    )
    
    calibrator.fit(X, y)
    
    calibrators[product] = calibrator
    
    print(
        f"{product}: calibrated successfully | "
        f"positives = {positives}"
    )

calibrators.keys()





validation_result = test_data.copy()
validation_result["calibrated_liabs"] = np.nan
validation_result["calibrated_assets"] = np.nan
validation_result["calibrated_fx"] = np.nan

for product, cfg in product_config.items():
    
    if product not in calibrators:
        continue
    
    score_col = cfg["score_col"]
    calibrated_col = cfg["calibrated_col"]
    
    mask = validation_result[score_col].notna()
    
    validation_result.loc[
        mask,
        calibrated_col
    ] = calibrators[product].predict_proba(
        validation_result.loc[mask, [score_col]]
    )[:, 1]

validation_result[
    [
        "IDENTIFYCODE",
        "score_month",
        "LIAB_PRIMARY",
        "ASSETS_PRIMARY",
        "FX_PRIMARY",
        "calibrated_liabs",
        "calibrated_assets",
        "calibrated_fx"
    ]
]





calibrated_cols = [
    "calibrated_liabs",
    "calibrated_assets",
    "calibrated_fx"
]

decision_scores = validation_result[
    calibrated_cols
].fillna(-np.inf)

validation_result["recommendation_score"] = (
    decision_scores.max(axis=1)
)

validation_result["recommended_product"] = (
    decision_scores.idxmax(axis=1)
    .map({
        "calibrated_liabs": "Liabilities",
        "calibrated_assets": "Assets",
        "calibrated_fx": "FX"
    })
)

no_recommendation_mask = validation_result[
    calibrated_cols
].isna().all(axis=1)

validation_result.loc[
    no_recommendation_mask,
    "recommended_product"
] = pd.NA

validation_result.loc[
    no_recommendation_mask,
    "recommendation_score"
] = np.nan


validation_result["recommended_raw_probability"] = np.select(
    [
        validation_result["recommended_product"].eq("Liabilities"),
        validation_result["recommended_product"].eq("Assets"),
        validation_result["recommended_product"].eq("FX")
    ],
    [
        validation_result["LIAB_PRIMARY"],
        validation_result["ASSETS_PRIMARY"],
        validation_result["FX_PRIMARY"]
    ],
    default=np.nan
)
validation_result["is_correct"] = (
    validation_result["recommended_product"]
    == validation_result["actual_product"]
)


















validation_result_clean = validation_result[
    (validation_result["actual_product"].notna())
    & (validation_result["n_actual_products"] == 1)
    & (validation_result["recommended_product"].notna())
].copy()
validation_result_clean["is_correct"].mean()




product_quality = (
    validation_result_clean
    .groupby("actual_product")
    .agg(
        clients=("IDENTIFYCODE", "count"),
        correct=("is_correct", "sum"),
        accuracy=("is_correct", "mean"),
        avg_recommendation_score=("recommendation_score", "mean")
    )
    .sort_values("clients", ascending=False)
)

product_quality


pd.crosstab(
    validation_result_clean["actual_product"],
    validation_result_clean["recommended_product"],
    normalize="index",
    margins=True
).round(3)