import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve


PRODUCTS = {
    "LIABILITIES": "LIAB_PRIMARY",
    "ASSETS": "ASSETS_PRIMARY",
    "FX": "FX_PRIMARY",
}

BETA = 0.5
N_SPLITS = 5       # validation ≈ 20%
RANDOM_STATE = 42

data = full_dataset.copy()


def normalize_target(value):
    if pd.isna(value) or value == "NONE":
        return "NONE"

    selected = {
        product.strip()
        for product in str(value).split(",")
    }

    # Фіксований порядок продуктів
    selected = [
        product
        for product in PRODUCTS
        if product in selected
    ]

    return ", ".join(selected) if selected else "NONE"


data["ACTUAL_PRODUCT"] = (
    data["ACTUAL_PRODUCT"]
    .apply(normalize_target)
)

# Окремий target для кожного продукту
for product in PRODUCTS:
    data[f"TARGET_{product}"] = (
        data["ACTUAL_PRODUCT"]
        .apply(
            lambda value: int(
                product in {
                    x.strip()
                    for x in value.split(",")
                }
            )
        )
    )

splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_idx, validation_idx = next(
    splitter.split(
        X=data,
        y=data["ACTUAL_PRODUCT"],
        groups=data["IDENTIFYCODE"],
    )
)

train_data = (
    data.iloc[train_idx]
    .reset_index(drop=True)
)

validation_data = (
    data.iloc[validation_idx]
    .reset_index(drop=True)
)

print("Train:", train_data.shape)
print("Validation:", validation_data.shape)

print(
    "Клієнтів одночасно у train і validation:",
    len(
        set(train_data["IDENTIFYCODE"])
        & set(validation_data["IDENTIFYCODE"])
    )
)

distribution = pd.concat(
    [
        data["ACTUAL_PRODUCT"]
        .value_counts(normalize=True)
        .rename("FULL"),

        train_data["ACTUAL_PRODUCT"]
        .value_counts(normalize=True)
        .rename("TRAIN"),

        validation_data["ACTUAL_PRODUCT"]
        .value_counts(normalize=True)
        .rename("VALIDATION"),
    ],
    axis=1,
).fillna(0)

display(distribution.round(4))
















def find_threshold(y_true, scores, beta=0.5):
    mask = scores.notna()

    y = y_true.loc[mask].to_numpy()
    score = scores.loc[mask].to_numpy()

    precision, recall, thresholds = precision_recall_curve(
        y,
        score,
    )

    if len(thresholds) == 0:
        return 1.0

    f_beta = (
        (1 + beta ** 2) * precision[:-1] * recall[:-1]
        / (
            beta ** 2 * precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    return float(thresholds[np.argmax(f_beta)])


thresholds = {}

for product, score_column in PRODUCTS.items():
    thresholds[product] = find_threshold(
        y_true=train_data[f"TARGET_{product}"],
        scores=train_data[score_column],
        beta=BETA,
    )

print(thresholds)

def make_recommendations(df, thresholds):
    recommendations = []

    for _, row in df.iterrows():
        selected = []

        for product, score_column in PRODUCTS.items():
            score = row[score_column]

            if (
                pd.notna(score)
                and score >= thresholds[product]
            ):
                selected.append(product)

        recommendations.append(
            ", ".join(selected)
            if selected
            else "NONE"
        )

    return recommendations


train_data["RECOMMENDED_PRODUCT"] = make_recommendations(
    train_data,
    thresholds,
)

validation_data["RECOMMENDED_PRODUCT"] = make_recommendations(
    validation_data,
    thresholds,
)

validation_data["RECOMMENDED_PRODUCT"].value_counts()