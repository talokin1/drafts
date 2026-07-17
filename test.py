from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss
)

target_cols = [f"TARGET_{p}" for p in PRODUCTS]
pred_cols = [f"PRED_{p}" for p in PRODUCTS]

y_true = validation[target_cols].astype(int).values
y_pred = validation[pred_cols].astype(int).values


# Метрики по кожному продукту
metrics = []

for i, product in enumerate(PRODUCTS):
    metrics.append({
        "PRODUCT": product,
        "PRECISION": precision_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        ),
        "RECALL": recall_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        ),
        "F1": f1_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        )
    })

metrics_df = pd.DataFrame(metrics)

display(metrics_df.round(3))




exact_match = (y_true == y_pred).all(axis=1).mean()

micro_f1 = f1_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

hamming = hamming_loss(y_true, y_pred)

print("Exact match:", round(exact_match, 3))
print("Micro F1:", round(micro_f1, 3))
print("Macro F1:", round(macro_f1, 3))
print("Hamming loss:", round(hamming, 3))




actual_none = y_true.sum(axis=1) == 0
pred_none = y_pred.sum(axis=1) == 0

print(
    "NONE precision:",
    round(
        precision_score(
            actual_none,
            pred_none,
            zero_division=0
        ),
        3
    )
)

print(
    "NONE recall:",
    round(
        recall_score(
            actual_none,
            pred_none,
            zero_division=0
        ),
        3
    )
)

print(
    "NONE F1:",
    round(
        f1_score(
            actual_none,
            pred_none,
            zero_division=0
        ),
        3
    )
)


comparison = pd.concat(
    [
        validation["ACTUAL_PRODUCT"]
        .value_counts()
        .rename("ACTUAL"),

        validation["RECOMMENDED_PRODUCT"]
        .value_counts()
        .rename("PREDICTED")
    ],
    axis=1
).fillna(0).astype(int)

display(comparison)