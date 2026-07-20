import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import MultiLabelBinarizer


PRODUCTS = {
    "LIABILITIES": "LIAB_PRIMARY",
    "ASSETS": "ASSETS_PRIMARY",
    "FX": "FX_PRIMARY"
}

LABELS = ["LIABILITIES", "ASSETS", "FX", "NOTHING_TO_DO"]


# 1. Підготовка
df = model_dataset.copy()

# Ненадійні записи не використовуємо для тюнінгу та перевірки
df = df[~df["RECORD_TYPE"].isin(["BAD_RECORD", "BAD"])].copy()


def parse_products(value):
    if pd.isna(value):
        return {"NOTHING_TO_DO"}

    products = {
        x.strip().upper()
        for x in str(value).split(",")
        if x.strip()
    }

    return products or {"NOTHING_TO_DO"}


df["TARGET_SET"] = df["ACTUAL_PRODUCT"].apply(parse_products)

# Канонічне представлення таргета для stratify
df["STRATIFY_TARGET"] = df["TARGET_SET"].apply(
    lambda x: ",".join(sorted(x))
)

# Рідкі комбінації стратифікуємо хоча б за типом запису
counts = df["STRATIFY_TARGET"].value_counts()
rare = df["STRATIFY_TARGET"].map(counts) < 5

df.loc[rare, "STRATIFY_TARGET"] = df.loc[rare, "RECORD_TYPE"]


train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["STRATIFY_TARGET"]
)


# 2. Підбір окремого threshold для кожного продукту
thresholds = {}

for product, score_col in PRODUCTS.items():
    y_true = train["TARGET_SET"].apply(lambda x: product in x).astype(int)
    scores = train[score_col].fillna(-1)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = (scores >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    thresholds[product] = best_threshold


# 3. Rule-based рекомендація
def make_predictions(data, thresholds, tie_delta):
    predictions = []

    for _, row in data.iterrows():
        margins = {}

        for product, score_col in PRODUCTS.items():
            score = row[score_col]
            threshold = thresholds[product]

            if pd.notna(score) and score >= threshold:
                # Нормалізоване перевищення власного threshold
                margins[product] = (
                    (score - threshold) / max(1 - threshold, 1e-6)
                )

        if not margins:
            predictions.append({"NOTHING_TO_DO"})
            continue

        ranked = sorted(
            margins,
            key=margins.get,
            reverse=True
        )

        result = {ranked[0]}

        if (
            len(ranked) > 1
            and margins[ranked[0]] - margins[ranked[1]] <= tie_delta
        ):
            result.add(ranked[1])

        predictions.append(result)

    return predictions


# 4. Підбір різниці для рекомендації двох продуктів
mlb = MultiLabelBinarizer(classes=LABELS)
mlb.fit([LABELS])

y_train = mlb.transform(train["TARGET_SET"])

best_delta = 0
best_macro_f1 = -1

for delta in np.arange(0, 0.31, 0.02):
    train_predictions = make_predictions(train, thresholds, delta)
    y_pred = mlb.transform(train_predictions)

    score = f1_score(
        y_train,
        y_pred,
        average="macro",
        zero_division=0
    )

    if score > best_macro_f1:
        best_macro_f1 = score
        best_delta = delta


# 5. Метрики на test
test_predictions = make_predictions(test, thresholds, best_delta)

y_true = mlb.transform(test["TARGET_SET"])
y_pred = mlb.transform(test_predictions)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)

metrics = pd.Series({
    "Exact match accuracy": accuracy_score(y_true, y_pred),

    "Hit rate": np.mean([
        bool(actual & predicted)
        for actual, predicted
        in zip(test["TARGET_SET"], test_predictions)
    ]),

    "Macro precision": precision_macro,
    "Macro recall": recall_macro,
    "Macro F1": f1_macro,

    "Positive hit rate": np.mean([
        bool(actual & predicted)
        for actual, predicted
        in zip(test["TARGET_SET"], test_predictions)
        if actual != {"NOTHING_TO_DO"}
    ]),

    "NOTHING_TO_DO accuracy": np.mean([
        predicted == {"NOTHING_TO_DO"}
        for actual, predicted
        in zip(test["TARGET_SET"], test_predictions)
        if actual == {"NOTHING_TO_DO"}
    ])
})


# Метрики окремо за продуктами
p, r, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    average=None,
    zero_division=0
)

product_metrics = pd.DataFrame({
    "Product": LABELS,
    "Precision": p,
    "Recall": r,
    "F1": f1,
    "Support": support
})

test_result = test.copy()
test_result["RECOMMENDED_PRODUCT"] = [
    ", ".join(sorted(x)) for x in test_predictions
]

print("Thresholds:", thresholds)
print("Tie delta:", round(best_delta, 2))

display(metrics.round(4))
display(product_metrics.round(4))