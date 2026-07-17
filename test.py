def make_recommendations(df, thresholds):
    def recommend_client(row):
        recommended = [
            product
            for product, score_column in PRODUCTS.items()
            if (
                pd.notna(row[score_column])
                and row[score_column] >= thresholds[product]
            )
        ]

        return ", ".join(recommended) if recommended else "NONE"

    return df.apply(recommend_client, axis=1)


validation_rule["RECOMMENDED_PRODUCT"] = make_recommendations(
    validation_rule,
    thresholds
)

validation_rule["RECOMMENDED_PRODUCT"].value_counts()







def to_product_set(value):
    if pd.isna(value) or value == "NONE":
        return set()

    return {
        product.strip()
        for product in value.split(",")
    }


validation_rule["CORRECT_RECOMMENDATION"] = [
    to_product_set(actual) == to_product_set(recommended)
    for actual, recommended in zip(
        validation_rule["ACTUAL_PRODUCT"],
        validation_rule["RECOMMENDED_PRODUCT"]
    )
]

print(
    "Exact match:",
    validation_rule["CORRECT_RECOMMENDATION"].mean()
)

for product in PRODUCTS:
    y_true = validation_rule[f"TARGET_{product}"]

    y_pred = (
        validation_rule["RECOMMENDED_PRODUCT"]
        .apply(lambda value: product in to_product_set(value))
        .astype(int)
    )

    print(
        product,
        "| precision:", round(
            precision_score(y_true, y_pred, zero_division=0), 3
        ),
        "| recall:", round(
            recall_score(y_true, y_pred, zero_division=0), 3
        ),
        "| f1:", round(
            f1_score(y_true, y_pred, zero_division=0), 3
        )
    )