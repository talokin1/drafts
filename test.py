from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report
)

def evaluate_model(name, y_true, probabilities, top_k=50):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    y_pred = probabilities.argmax(axis=1)

    top_k = min(top_k, len(y_true))
    top_idx = np.argsort(probabilities[:, 2])[::-1][:top_k]

    hnwi_total = np.sum(y_true == 2)
    hnwi_hits = np.sum(y_true[top_idx] == 2)

    one_hot = np.eye(3)[y_true]
    brier = np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))

    metrics = pd.Series({
        "macro_f1": f1_score(
            y_true, y_pred, average="macro"
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "quadratic_kappa": cohen_kappa_score(
            y_true, y_pred, weights="quadratic"
        ),
        "ordinal_mae": mean_absolute_error(
            y_true, y_pred
        ),
        "HNWI_average_precision": average_precision_score(
            y_true == 2, probabilities[:, 2]
        ),
        f"precision_at_{top_k}": hnwi_hits / top_k,
        f"recall_at_{top_k}": (
            hnwi_hits / hnwi_total if hnwi_total else np.nan
        ),
        "HNWI_to_MASS_rate": (
            np.mean(y_pred[y_true == 2] == 0)
            if hnwi_total else np.nan
        ),
        "log_loss": log_loss(
            y_true, probabilities, labels=[0, 1, 2]
        ),
        "multiclass_brier": brier
    })

    print(f"\n{name}")
    display(metrics.round(4))

    print("\nConfusion matrix:")
    display(pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
        index=[f"TRUE_{x}" for x in TARGET_NAMES],
        columns=[f"PRED_{x}" for x in TARGET_NAMES]
    ))

    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        digits=3,
        zero_division=0
    ))

    return metrics