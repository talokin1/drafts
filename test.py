import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)


PRODUCTS = {
    "Liabilities": {
        "score": "LIAB_PRIMARY",
        "target": "NEW_LIABILITIES",
        "rec": "REC_LIABILITIES"
    },
    "Assets": {
        "score": "ASSETS_PRIMARY",
        "target": "NEW_ASSETS",
        "rec": "REC_ASSETS"
    },
    "FX": {
        "score": "FX_PRIMARY",
        "target": "NEW_FX",
        "rec": "REC_FX"
    }
}

target_cols = [
    config["target"]
    for config in PRODUCTS.values()
]

data = (
    validation_model
    .dropna(subset=target_cols)
    .copy()
)

months = sorted(
    data["score_month"].unique(),
    key=lambda month: pd.Period(
        month.replace("_", "-"),
        freq="M"
    )
)

if len(months) < 2:
    raise ValueError("Потрібно мінімум два місяці")

split_index = max(1, int(len(months) * 0.8))

if split_index == len(months):
    split_index = len(months) - 1

tune_months = months[:split_index]
test_months = months[split_index:]

validation_tune = data[
    data["score_month"].isin(tune_months)
].copy()

validation_test = data[
    data["score_month"].isin(test_months)
].copy()

print("Tune:", tune_months)
print("Test:", test_months)















def find_pr_threshold(df, score_col, target_col):
    scores = pd.to_numeric(
        df[score_col],
        errors="coerce"
    )

    mask = scores.notna() & df[target_col].notna()

    scores = scores[mask]
    target = df.loc[mask, target_col].astype(bool)

    if scores.empty:
        raise ValueError(
            f"Немає значень {score_col} у tune-вибірці"
        )

    if target.sum() == 0:
        raise ValueError(
            f"Немає позитивних {target_col} у tune-вибірці"
        )

    precision, recall, thresholds = (
        precision_recall_curve(target, scores)
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    return float(
        thresholds[np.argmax(f1)]
    )


initial_thresholds = {
    product: find_pr_threshold(
        validation_tune,
        config["score"],
        config["target"]
    )
    for product, config in PRODUCTS.items()
}

initial_thresholds












def threshold_objective(df, thresholds):
    actual_arrays = []
    predicted_arrays = []
    f1_values = []

    for product, config in PRODUCTS.items():
        actual = (
            df[config["target"]]
            .astype(bool)
            .to_numpy()
        )

        predicted = (
            pd.to_numeric(
                df[config["score"]],
                errors="coerce"
            )
            .ge(thresholds[product])
            .fillna(False)
            .to_numpy()
        )

        actual_arrays.append(actual)
        predicted_arrays.append(predicted)

        f1_values.append(
            f1_score(
                actual,
                predicted,
                zero_division=0
            )
        )

    actual_matrix = np.column_stack(actual_arrays)
    predicted_matrix = np.column_stack(
        predicted_arrays
    )

    actual_none = ~actual_matrix.any(axis=1)
    predicted_none = ~predicted_matrix.any(axis=1)

    f1_none = f1_score(
        actual_none,
        predicted_none,
        zero_division=0
    )

    return np.mean(
        f1_values + [f1_none]
    )


def optimize_thresholds(
    df,
    initial_thresholds,
    step=0.01,
    max_iterations=10
):
    thresholds = initial_thresholds.copy()
    best_score = threshold_objective(
        df,
        thresholds
    )

    grid = np.round(
        np.arange(0, 1 + step, step),
        4
    )

    for _ in range(max_iterations):
        improved = False

        for product in PRODUCTS:
            best_product_threshold = thresholds[product]
            best_product_score = best_score

            for candidate_threshold in grid:
                candidate = thresholds.copy()
                candidate[product] = float(
                    candidate_threshold
                )

                candidate_score = threshold_objective(
                    df,
                    candidate
                )

                if candidate_score > best_product_score:
                    best_product_score = candidate_score
                    best_product_threshold = float(
                        candidate_threshold
                    )

            if best_product_score > best_score:
                thresholds[product] = (
                    best_product_threshold
                )

                best_score = best_product_score
                improved = True

        if not improved:
            break

    return thresholds, best_score


optimized_thresholds, tune_objective = (
    optimize_thresholds(
        validation_tune,
        initial_thresholds
    )
)

threshold_results = pd.DataFrame({
    "product": list(PRODUCTS),
    "initial_threshold": [
        initial_thresholds[p]
        for p in PRODUCTS
    ],
    "optimized_threshold": [
        optimized_thresholds[p]
        for p in PRODUCTS
    ]
})

threshold_results












def apply_thresholds(df, thresholds):
    result = df.copy()

    for product, config in PRODUCTS.items():
        result[config["rec"]] = (
            pd.to_numeric(
                result[config["score"]],
                errors="coerce"
            )
            .ge(thresholds[product])
            .fillna(False)
        )

    result["recommended_product"] = result.apply(
        lambda row: ", ".join(
            product
            for product, config in PRODUCTS.items()
            if row[config["rec"]]
        ) or "None",
        axis=1
    )

    return result


test_predictions = apply_thresholds(
    validation_test,
    optimized_thresholds
)


product_metrics = []

for product, config in PRODUCTS.items():
    actual = test_predictions[
        config["target"]
    ].astype(bool)

    predicted = test_predictions[
        config["rec"]
    ].astype(bool)

    product_metrics.append({
        "product": product,
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        ),
        "actual": int(actual.sum()),
        "recommended": int(predicted.sum())
    })

product_metrics = pd.DataFrame(product_metrics)

product_metrics













actual_matrix = test_predictions[
    target_cols
].astype(bool).to_numpy()

rec_cols = [
    config["rec"]
    for config in PRODUCTS.values()
]

predicted_matrix = test_predictions[
    rec_cols
].astype(bool).to_numpy()

actual_none = ~actual_matrix.any(axis=1)
predicted_none = ~predicted_matrix.any(axis=1)

test_summary = pd.Series({
    "tune_objective": tune_objective,

    "exact_accuracy":
        (actual_matrix == predicted_matrix)
        .all(axis=1)
        .mean(),

    "hit_rate_engaged":
        (
            (actual_matrix & predicted_matrix)
            .any(axis=1)[~actual_none]
            .mean()
        ),

    "correct_none_rate":
        predicted_none[actual_none].mean(),

    "f1_none":
        f1_score(
            actual_none,
            predicted_none,
            zero_division=0
        ),

    "macro_f1_products":
        product_metrics["f1"].mean()
})

test_summary