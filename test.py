import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)


PRODUCTS = {
    "LIABILITIES": "LIAB_PRIMARY",
    "ASSETS": "ASSETS_PRIMARY",
    "FX": "FX_PRIMARY"
}

BETA = 0.5


# 1. Створюємо окремий бінарний target для кожного продукту
def add_targets(df):
    df = df.copy()

    for product in PRODUCTS:
        df[f"TARGET_{product}"] = (
            df["ACTUAL_PRODUCT"]
            .fillna("NONE")
            .str.split(",")
            .apply(lambda values: product in [x.strip() for x in values])
            .astype(int)
        )

    return df


train_rule = add_targets(train_data)
validation_rule = add_targets(validation_data)


# 2. Підбираємо threshold для одного продукту
def find_threshold(y_true, scores, beta=0.5):
    mask = scores.notna()

    y_true = y_true[mask].values
    scores = scores[mask].values

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores
    )

    f_beta = (
        (1 + beta**2) * precision[:-1] * recall[:-1]
        / (
            beta**2 * precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    return float(thresholds[np.argmax(f_beta)])


thresholds = {
    product: find_threshold(
        train_rule[f"TARGET_{product}"],
        train_rule[score_column],
        beta=BETA
    )
    for product, score_column in PRODUCTS.items()
}

print("Thresholds:")
print(thresholds)