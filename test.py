import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import MultiLabelBinarizer


PRODUCTS = {
    "LIABILITIES": "LIAB_PRIMARY",
    "ASSETS": "ASSETS_PRIMARY",
    "FX": "FX_PRIMARY"
}

PRODUCT_NAMES = list(PRODUCTS)
MIN_NOTHING_RECALL = 0.93


# 1. Робимо скори міжпродуктово порівнюваними через percentile
score_distributions = {
    product: np.sort(
        pd.to_numeric(train[column], errors="coerce").dropna()
    )
    for product, column in PRODUCTS.items()
}


def to_percentile(values, distribution):
    values = pd.to_numeric(values, errors="coerce").fillna(-np.inf)
    return np.searchsorted(distribution, values, side="right") / max(len(distribution), 1)


for data in [train, test]:
    for product, column in PRODUCTS.items():
        data[f"{product}_PCT"] = to_percentile(
            data[column],
            score_distributions[product]
        )

    data["ACTION_SCORE"] = data[
        [f"{p}_PCT" for p in PRODUCT_NAMES]
    ].max(axis=1)


# 2. Етап ACTION / NOTHING_TO_DO
y_action_train = train["TARGET_SET"].apply(
    lambda x: x != {"NOTHING_TO_DO"}
).astype(int)

best_gate = None

for threshold in np.linspace(0.50, 0.99, 100):
    pred = (train["ACTION_SCORE"] >= threshold).astype(int)

    nothing_recall = (
        ((pred == 0) & (y_action_train == 0)).sum()
        / max((y_action_train == 0).sum(), 1)
    )

    action_f1 = f1_score(y_action_train, pred, zero_division=0)

    if nothing_recall >= MIN_NOTHING_RECALL:
        if best_gate is None or action_f1 > best_gate[0]:
            best_gate = (action_f1, threshold, nothing_recall)

# Якщо 93% недосяжні
if best_gate is None:
    best_gate = max(
        (
            f1_score(
                y_action_train,
                (train["ACTION_SCORE"] >= t).astype(int),
                zero_division=0
            ),
            t,
            0
        )
        for t in np.linspace(0.50, 0.99, 100)
    )

gate_threshold = best_gate[1]


# 3. Тюнінг продуктів тільки серед ACTION-клієнтів
stage2_train = train[
    (y_action_train == 1)
    & (train["ACTION_SCORE"] >= gate_threshold)
].copy()

product_thresholds = {}

for product in PRODUCT_NAMES:
    y_true = stage2_train["TARGET_SET"].apply(
        lambda x: product in x
    ).astype(int)

    scores = stage2_train[f"{product}_PCT"]

    product_thresholds[product] = max(
        np.linspace(0.05, 0.95, 91),
        key=lambda t: f1_score(
            y_true,
            (scores >= t).astype(int),
            zero_division=0
        )
    )


# 4. Фінальна рекомендація
def predict(data, tie_delta):
    predictions = []

    for _, row in data.iterrows():

        if row["ACTION_SCORE"] < gate_threshold:
            predictions.append({"NOTHING_TO_DO"})
            continue

        ranked = sorted(
            PRODUCT_NAMES,
            key=lambda p: row[f"{p}_PCT"],
            reverse=True
        )

        passed = [
            p for p in ranked
            if row[f"{p}_PCT"] >= product_thresholds[p]
        ]

        # Gate сказав ACTION, тому хоча б top-1 має бути обраний
        selected = passed if passed else [ranked[0]]
        result = {selected[0]}

        if (
            len(selected) > 1
            and row[f"{selected[0]}_PCT"] - row[f"{selected[1]}_PCT"]
            <= tie_delta
        ):
            result.add(selected[1])

        predictions.append(result)

    return predictions


# 5. Тюнінг другого продукту
mlb_products = MultiLabelBinarizer(classes=PRODUCT_NAMES)
mlb_products.fit([PRODUCT_NAMES])

y_stage2 = mlb_products.transform(stage2_train["TARGET_SET"])

best_delta = max(
    np.arange(0, 0.31, 0.02),
    key=lambda delta: f1_score(
        y_stage2,
        mlb_products.transform(
            [
                pred & set(PRODUCT_NAMES)
                for pred in predict(stage2_train, delta)
            ]
        ),
        average="macro",
        zero_division=0
    )
)


# 6. Метрики на test
test_predictions = predict(test, best_delta)

test["RECOMMENDED_PRODUCT"] = [
    ", ".join(sorted(x)) for x in test_predictions
]

y_action_test = test["TARGET_SET"].apply(
    lambda x: x != {"NOTHING_TO_DO"}
).astype(int)

action_pred = (
    test["ACTION_SCORE"] >= gate_threshold
).astype(int)

gate_precision, gate_recall, gate_f1, _ = (
    precision_recall_fscore_support(
        y_action_test,
        action_pred,
        average="binary",
        zero_division=0
    )
)

nothing_recall = (
    ((action_pred == 0) & (y_action_test == 0)).sum()
    / max((y_action_test == 0).sum(), 1)
)

LABELS = PRODUCT_NAMES + ["NOTHING_TO_DO"]

mlb = MultiLabelBinarizer(classes=LABELS)
mlb.fit([LABELS])

y_true = mlb.transform(test["TARGET_SET"])
y_pred = mlb.transform(test_predictions)

p, r, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    average=None,
    zero_division=0
)

metrics = pd.Series({
    "Exact match accuracy": accuracy_score(y_true, y_pred),
    "Hit rate": np.mean([
        bool(actual & predicted)
        for actual, predicted
        in zip(test["TARGET_SET"], test_predictions)
    ]),
    "ACTION precision": gate_precision,
    "ACTION recall": gate_recall,
    "ACTION F1": gate_f1,
    "NOTHING recall": nothing_recall
})

product_metrics = pd.DataFrame({
    "Product": LABELS,
    "Precision": p,
    "Recall": r,
    "F1": f1,
    "Support": support
})

print("Gate threshold:", round(gate_threshold, 3))
print("Product thresholds:", product_thresholds)
print("Tie delta:", round(best_delta, 2))

display(metrics.round(4))
display(product_metrics.round(4))