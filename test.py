PRODUCT_NAMES = ["LIABILITIES", "ASSETS", "FX"]
PCT_COLS = [f"{p}_PCT" for p in PRODUCT_NAMES]

MIN_NOTHING_RECALL = 0.88
N_TRIALS = 4000
rng = np.random.default_rng(42)


def prepare_arrays(data):
    scores = data[PCT_COLS].to_numpy(float)

    targets = np.column_stack([
        data["TARGET_SET"].apply(lambda x: p in x)
        for p in PRODUCT_NAMES
    ]).astype(int)

    return scores, targets


def predict_array(scores, thresholds, gate, tie_delta):
    margins = (scores - thresholds) / np.maximum(1 - thresholds, 1e-6)
    margins = np.where(margins >= 0, margins, -np.inf)

    order = np.argsort(margins, axis=1)[:, ::-1]
    rows = np.arange(len(scores))

    top1 = order[:, 0]
    top2 = order[:, 1]

    top1_margin = margins[rows, top1]
    top2_margin = margins[rows, top2]

    action = top1_margin >= gate

    pred = np.zeros_like(scores, dtype=int)
    pred[rows[action], top1[action]] = 1

    add_second = (
        action
        & np.isfinite(top2_margin)
        & ((top1_margin - top2_margin) <= tie_delta)
    )

    pred[rows[add_second], top2[add_second]] = 1

    return pred


def evaluate(y_true, y_pred):
    tp = (y_true & y_pred).sum(axis=0)
    fp = ((1 - y_true) & y_pred).sum(axis=0)
    fn = (y_true & (1 - y_pred)).sum(axis=0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

    actual_action = y_true.any(axis=1)
    pred_action = y_pred.any(axis=1)

    action_tp = (actual_action & pred_action).sum()
    action_fp = (~actual_action & pred_action).sum()
    action_fn = (actual_action & ~pred_action).sum()

    action_precision = action_tp / max(action_tp + action_fp, 1)
    action_recall = action_tp / max(action_tp + action_fn, 1)
    action_f1 = (
        2 * action_precision * action_recall
        / max(action_precision + action_recall, 1e-9)
    )

    nothing_recall = np.mean(~pred_action[~actual_action])

    positive_hit = np.mean(
        (y_true[actual_action] & y_pred[actual_action]).any(axis=1)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": f1.mean(),
        "action_precision": action_precision,
        "action_recall": action_recall,
        "action_f1": action_f1,
        "nothing_recall": nothing_recall,
        "positive_hit": positive_hit
    }


# Тюнінг усіх параметрів одночасно
X_train, y_train = prepare_arrays(train)

best = None

for _ in range(N_TRIALS):
    thresholds = rng.uniform(0.01, 0.99, 3)
    gate = rng.uniform(0, 0.50)
    tie_delta = rng.uniform(0, 0.25)

    pred = predict_array(X_train, thresholds, gate, tie_delta)
    m = evaluate(y_train, pred)

    if m["nothing_recall"] < MIN_NOTHING_RECALL:
        continue

    objective = (
        0.45 * m["macro_f1"]
        + 0.25 * m["action_f1"]
        + 0.20 * m["positive_hit"]
        + 0.10 * m["nothing_recall"]
    )

    if best is None or objective > best["objective"]:
        best = {
            "objective": objective,
            "thresholds": thresholds,
            "gate": gate,
            "tie_delta": tie_delta
        }


# Фінальна перевірка на test
X_test, y_test = prepare_arrays(test)

test_pred = predict_array(
    X_test,
    best["thresholds"],
    best["gate"],
    best["tie_delta"]
)

m = evaluate(y_test, test_pred)

product_metrics = pd.DataFrame({
    "Product": PRODUCT_NAMES,
    "Precision": m["precision"],
    "Recall": m["recall"],
    "F1": m["f1"],
    "Support": y_test.sum(axis=0)
})

test["RECOMMENDED_PRODUCT"] = [
    ", ".join(np.array(PRODUCT_NAMES)[row.astype(bool)])
    if row.any() else "NOTHING_TO_DO"
    for row in test_pred
]

metrics = pd.Series({
    "ACTION precision": m["action_precision"],
    "ACTION recall": m["action_recall"],
    "ACTION F1": m["action_f1"],
    "Positive hit rate": m["positive_hit"],
    "NOTHING recall": m["nothing_recall"],
    "Product Macro F1": m["macro_f1"]
})

print("Gate:", round(best["gate"], 3))
print(
    "Thresholds:",
    dict(zip(PRODUCT_NAMES, best["thresholds"].round(3)))
)
print("Tie delta:", round(best["tie_delta"], 3))

display(metrics.round(4))
display(product_metrics.round(4))