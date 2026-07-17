import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve


FEATURES = [
    "LIAB_PRIMARY",
    "ASSETS_PRIMARY",
    "FX_PRIMARY"
]

PRODUCTS = ["LIABILITIES", "ASSETS", "FX"]

data = full_dataset.copy()


# ============================================================
# 1. MULTI-LABEL TARGET
# ============================================================

def parse_products(value):
    if pd.isna(value) or value == "NONE":
        return set()

    return {
        product.strip()
        for product in str(value).split(",")
    }


for product in PRODUCTS:
    data[f"TARGET_{product}"] = (
        data["ACTUAL_PRODUCT"]
        .apply(lambda x: int(product in parse_products(x)))
    )


# ============================================================
# 2. РЕПРЕЗЕНТАТИВНИЙ GROUP SPLIT
# ============================================================

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, valid_idx = next(
    splitter.split(
        data[FEATURES],
        data["ACTUAL_PRODUCT"],
        groups=data["IDENTIFYCODE"]
    )
)

train = data.iloc[train_idx].copy()
validation = data.iloc[valid_idx].copy()


# ============================================================
# 3. ТРИ НЕЗАЛЕЖНІ МОДЕЛІ
# ============================================================

models = {}
thresholds = {}

for product in PRODUCTS:

    y_train = train[f"TARGET_{product}"]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )

    # OOF-прогнози лише для вибору threshold
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    oof_probability = cross_val_predict(
        model,
        train[FEATURES],
        y_train,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    precision, recall, threshold_values = precision_recall_curve(
        y_train,
        oof_probability
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    threshold = threshold_values[np.argmax(f1)]

    model.fit(train[FEATURES], y_train)

    models[product] = model
    thresholds[product] = threshold


print("Thresholds:", thresholds)


for product in PRODUCTS:

    validation[f"PROB_{product}"] = (
        models[product]
        .predict_proba(validation[FEATURES])[:, 1]
    )

    validation[f"PRED_{product}"] = (
        validation[f"PROB_{product}"]
        >= thresholds[product]
    )


def get_recommendation(row):
    recommended = [
        product
        for product in PRODUCTS
        if row[f"PRED_{product}"]
    ]

    return ", ".join(recommended) if recommended else "NONE"


validation["RECOMMENDED_PRODUCT"] = validation.apply(
    get_recommendation,
    axis=1
)

validation["RECOMMENDED_PRODUCT"].value_counts()