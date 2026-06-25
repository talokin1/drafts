evaluation = validation_result_clean.copy()

always_liabs_accuracy = (
    evaluation["actual_product"]
    .eq("Liabilities")
    .mean()
)

always_assets_accuracy = (
    evaluation["actual_product"]
    .eq("Assets")
    .mean()
)

calibrated_accuracy = evaluation["is_correct"].mean()

pd.DataFrame({
    "strategy": [
        "Always Liabilities",
        "Always Assets",
        "Calibrated rule-based"
    ],
    "accuracy": [
        always_liabs_accuracy,
        always_assets_accuracy,
        calibrated_accuracy
    ]
}).sort_values("accuracy", ascending=False)




(
    validation_result_clean
    .groupby("actual_product")
    .agg(
        clients=("IDENTIFYCODE", "count"),
        
        avg_calibrated_liabs=("calibrated_liabs", "mean"),
        avg_calibrated_assets=("calibrated_assets", "mean"),
        
        median_calibrated_liabs=("calibrated_liabs", "median"),
        median_calibrated_assets=("calibrated_assets", "median")
    )
    .round(4)
)




validation_result_clean.loc[
    (
        validation_result_clean["actual_product"].eq("Assets")
        & ~validation_result_clean["is_correct"]
    ),
    [
        "score_month",
        "IDENTIFYCODE",
        "LIAB_PRIMARY",
        "ASSETS_PRIMARY",
        "calibrated_liabs",
        "calibrated_assets",
        "recommendation_score",
        "recommended_product",
        "actual_product"
    ]
]