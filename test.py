validation_for_merge = validation_all[
    [
        "IDENTIFYCODE",
        "was_engaged",
        "n_actual_products",
        "actual_products_list",
        "actual_product",
        "NEW_LIABILITIES",
        "NEW_ASSETS",
        "NEW_FX"
    ]
].copy()

evaluation_df = rec_eval.merge(
    validation_for_merge,
    on="IDENTIFYCODE",
    how="left"
)

# Якщо клієнта не знайшли у validation_all,
# не вважаємо це автоматично відсутністю залучення.
evaluation_df["in_validation"] = evaluation_df["was_engaged"].notna()

evaluation_df["was_engaged"] = (
    evaluation_df["was_engaged"]
    .fillna(False)
    .astype(bool)
)

evaluation_df["n_actual_products"] = (
    evaluation_df["n_actual_products"]
    .fillna(0)
    .astype(int)
)

evaluation_df["actual_products_list"] = evaluation_df["actual_products_list"].apply(
    lambda x: x if isinstance(x, list) else []
)

# Чи рекомендований продукт входить у фактичні залучення клієнта
evaluation_df["recommendation_hit"] = evaluation_df.apply(
    lambda row: (
        row["recommended_product_clean"] in row["actual_products_list"]
    ),
    axis=1
)

# Для наочності
evaluation_df["recommendation_result"] = np.select(
    [
        ~evaluation_df["in_validation"],
        evaluation_df["recommendation_hit"],
        evaluation_df["was_engaged"] & ~evaluation_df["recommendation_hit"],
        ~evaluation_df["was_engaged"]
    ],
    [
        "No validation data",
        "Hit",
        "Engaged in another product",
        "No engagement"
    ],
    default="Unknown"
)

display(
    evaluation_df[
        [
            "IDENTIFYCODE",
            "recommended_product_clean",
            "recommendation_score",
            "was_engaged",
            "actual_product",
            "n_actual_products",
            "recommendation_hit",
            "recommendation_result"
        ]
    ].head(20)
)












# =========================================================
# 7. Загальні метрики
# =========================================================

eval_base = evaluation_df.loc[
    evaluation_df["in_validation"]
].copy()

n_recommended = len(eval_base)
n_engaged_any = eval_base["was_engaged"].sum()
n_hits = eval_base["recommendation_hit"].sum()

metrics_summary = pd.DataFrame(
    {
        "metric": [
            "Recommended clients in validation",
            "Clients engaged in any product",
            "Recommendation hits",
            "Overall recommendation conversion",
            "Precision among engaged clients",
            "Share of engaged clients captured by recommendation",
            "Share engaged in another product",
            "Share without engagement"
        ],
        "value": [
            n_recommended,
            n_engaged_any,
            n_hits,

            # З усіх рекомендованих: скільки реально купили саме рекомендований продукт
            n_hits / n_recommended if n_recommended > 0 else np.nan,

            # Серед тих, хто взагалі щось купив:
            # скільки купили саме рекомендований продукт
            n_hits / n_engaged_any if n_engaged_any > 0 else np.nan,

            # Та сама логіка для top-1 рекомендації
            n_hits / n_engaged_any if n_engaged_any > 0 else np.nan,

            (
                (
                    (eval_base["was_engaged"])
                    & (~eval_base["recommendation_hit"])
                ).mean()
                if n_recommended > 0 else np.nan
            ),

            (
                (~eval_base["was_engaged"]).mean()
                if n_recommended > 0 else np.nan
            )
        ]
    }
)

display(metrics_summary)














# =========================================================
# 8. Метрики за рекомендованим продуктом
# =========================================================

product_metrics = (
    eval_base
    .groupby("recommended_product_clean", dropna=False)
    .agg(
        recommended_clients=("IDENTIFYCODE", "count"),
        engaged_any_product=("was_engaged", "sum"),
        recommendation_hits=("recommendation_hit", "sum"),
        avg_recommendation_score=("recommendation_score", "mean")
    )
    .reset_index()
)

product_metrics["conversion_to_recommended_product"] = (
    product_metrics["recommendation_hits"]
    / product_metrics["recommended_clients"]
)

product_metrics["precision_among_engaged"] = (
    product_metrics["recommendation_hits"]
    / product_metrics["engaged_any_product"]
)

product_metrics = product_metrics.sort_values(
    "recommended_clients",
    ascending=False
)

display(product_metrics)



















# =========================================================
# 9. Матриця: що рекомендували vs що фактично взяли
# =========================================================

actual_products_exploded = (
    eval_base[
        [
            "IDENTIFYCODE",
            "recommended_product_clean",
            "actual_products_list"
        ]
    ]
    .explode("actual_products_list")
    .rename(columns={"actual_products_list": "actual_product_single"})
)

recommendation_vs_actual = pd.crosstab(
    actual_products_exploded["recommended_product_clean"],
    actual_products_exploded["actual_product_single"],
    margins=True
)

display(recommendation_vs_actual)