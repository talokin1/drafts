import pandas as pd


def product_set(value):
    if pd.isna(value):
        return set()

    value = str(value).strip()

    if value.lower() in {"none", "nan", ""}:
        return set()

    return {
        product.strip()
        for product in value.split(",")
        if product.strip()
    }


metrics_df = validation_model.copy()

metrics_df["actual_set"] = (
    metrics_df["actual_product"].apply(product_set)
)

metrics_df["recommended_set"] = (
    metrics_df["recommended_product"].apply(product_set)
)


# Повний збіг наборів продуктів
metrics_df["exact_match"] = metrics_df.apply(
    lambda row:
        row["actual_set"] == row["recommended_set"],
    axis=1
)

# Хоча б один правильно рекомендований продукт
metrics_df["has_hit"] = metrics_df.apply(
    lambda row: (
        bool(
            row["actual_set"]
            & row["recommended_set"]
        )
        if row["actual_set"]
        else not row["recommended_set"]
    ),
    axis=1
)

# Jaccard для частково правильних рекомендацій
metrics_df["jaccard"] = metrics_df.apply(
    lambda row: (
        len(
            row["actual_set"]
            & row["recommended_set"]
        )
        / len(
            row["actual_set"]
            | row["recommended_set"]
        )
        if row["actual_set"] | row["recommended_set"]
        else 1.0
    ),
    axis=1
)







actual_positive = metrics_df[
    metrics_df["actual_set"].map(bool)
]

actual_none = metrics_df[
    ~metrics_df["actual_set"].map(bool)
]

summary = pd.Series({
    "exact_accuracy_all":
        metrics_df["exact_match"].mean(),

    "exact_accuracy_engaged":
        actual_positive["exact_match"].mean(),

    "hit_rate_engaged":
        actual_positive["has_hit"].mean(),

    "correct_none_rate":
        actual_none["exact_match"].mean(),

    "mean_jaccard":
        metrics_df["jaccard"].mean()
})

summary



product_metrics = []

for product in ["Liabilities", "Assets", "FX"]:
    actual = metrics_df["actual_set"].apply(
        lambda products: product in products
    )

    predicted = metrics_df["recommended_set"].apply(
        lambda products: product in products
    )

    tp = (actual & predicted).sum()
    fp = (~actual & predicted).sum()
    fn = (actual & ~predicted).sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    product_metrics.append({
        "product": product,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "actual": int(actual.sum()),
        "recommended": int(predicted.sum())
    })

product_metrics = pd.DataFrame(product_metrics)

product_metrics