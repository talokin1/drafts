validation_result = validation_dataset.copy()

# Перейменовуємо скори у формат рекомендаційки
validation_result = validation_result.rename(
    columns={
        "LIAB_PRIMARY": "p_liabs",
        "ASSETS_PRIMARY": "p_assets",
        "FX_PRIMARY": "p_fx"
    }
)

thresholds = {
    "p_liabs": 0.30,
    "p_assets": 0.30,
    "p_fx": 0.59
}


# Рахуємо normalized score окремо для кожного продукту.
# NaN зберігається: продукт недоступний для рекомендації.

validation_result["score_liabs"] = np.where(
    validation_result["p_liabs"].notna(),
    scale_by_threshold(
        validation_result["p_liabs"],
        thresholds["p_liabs"]
    ),
    np.nan
)

validation_result["score_assets"] = np.where(
    validation_result["p_assets"].notna(),
    scale_by_threshold(
        validation_result["p_assets"],
        thresholds["p_assets"]
    ),
    np.nan
)

validation_result["score_fx"] = np.where(
    validation_result["p_fx"].notna(),
    scale_by_threshold(
        validation_result["p_fx"],
        thresholds["p_fx"]
    ),
    np.nan
)



score_cols = ["score_liabs", "score_assets", "score_fx"]

# Для вибору максимуму недоступні продукти тимчасово ставимо в -inf.
# Це не дає, наприклад, Liabilities виграти лише тому,
# що всі скори були заповнені нулями.

decision_scores = validation_result[score_cols].fillna(-np.inf)

validation_result["recommendation_score"] = (
    decision_scores.max(axis=1)
)

validation_result["recommended_product"] = (
    decision_scores.idxmax(axis=1)
    .map({
        "score_liabs": "Liabilities",
        "score_assets": "Assets",
        "score_fx": "FX"
    })
)

# Якщо для клієнта взагалі не було жодного доступного продукту
validation_result.loc[
    validation_result[score_cols].isna().all(axis=1),
    ["recommended_product", "recommendation_score"]
] = [pd.NA, np.nan]



# Сира ймовірність саме рекомендованого продукту

validation_result["recommended_raw_probability"] = np.select(
    [
        validation_result["recommended_product"].eq("Liabilities"),
        validation_result["recommended_product"].eq("Assets"),
        validation_result["recommended_product"].eq("FX")
    ],
    [
        validation_result["p_liabs"],
        validation_result["p_assets"],
        validation_result["p_fx"]
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







month_quality = (
    validation_result_clean
    .groupby("score_month")
    .agg(
        clients=("IDENTIFYCODE", "count"),
        correct=("is_correct", "sum"),
        accuracy=("is_correct", "mean")
    )
    .sort_index()
)

month_quality












validation_result_clean[
    validation_result_clean["is_correct"] == False
][
    [
        "score_month",
        "IDENTIFYCODE",
        "actual_product",
        "recommended_product",
        "p_liabs",
        "p_assets",
        "p_fx",
        "score_liabs",
        "score_assets",
        "score_fx",
        "recommendation_score"
    ]
].head(30)





pd.crosstab(
    validation_result_clean["actual_product"],
    validation_result_clean["recommended_product"],
    margins=True,
    normalize="index"
).round(3)