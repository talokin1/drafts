import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, confusion_matrix, classification_report

MIN_RECALL = 0.15
TOP_K = 50
SEEDS = (42, 52, 62)
OUTER_SPLITS = 4
INNER_SPLITS = 3

# Додай сюди всі поля, які прямо або опосередковано використовувалися для створення SEGMENT
LEAKAGE_COLS = ["MOBILEPHONE", "group_id", "SEGMENT", "CONTRAGENTID", "is_hnwi", "IS_HNWI", "BALANCE"]

data = client_df.reset_index(drop=True).copy()
groups = data["group_id"].astype(str)

# MASS = 0, PREM = 0, HNWI = 1
y = data["SEGMENT"].eq("HNWI").astype(int)
X = data.drop(columns=LEAKAGE_COLS, errors="ignore").replace([np.inf, -np.inf], np.nan)

cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
for col in cat_cols: X[col] = X[col].astype("string").fillna("Missing").astype(str)

print(pd.crosstab(data["SEGMENT"], y, rownames=["SEGMENT"], colnames=["TARGET"]))


def create_model(seed):
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=700,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=10,
        random_strength=1,
        auto_class_weights="SqrtBalanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False
    )


def binary_oof(X, y, groups, n_splits):
    oof_probability = np.zeros(len(X))

    for seed in SEEDS:
        seed_probability = np.zeros(len(X))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
            model = create_model(seed + fold)
            model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_cols)
            seed_probability[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

        oof_probability += seed_probability / len(SEEDS)

    return oof_probability


def choose_threshold(y_true, probability, min_recall=MIN_RECALL):
    best = None

    for threshold in np.unique(probability):
        prediction = (probability >= threshold).astype(int)

        if prediction.sum() == 0:
            continue

        recall = recall_score(y_true, prediction, zero_division=0)

        if recall < min_recall:
            continue

        precision = precision_score(y_true, prediction, zero_division=0)
        f05 = fbeta_score(y_true, prediction, beta=0.5, zero_division=0)
        key = (precision, f05, recall, -prediction.sum())

        if best is None or key > best[0]:
            best = (key, threshold)

    if best is None:
        raise ValueError("Не знайдено threshold із заданим minimum recall")

    return best[1]


outer_probability = np.zeros(len(X))
outer_prediction = np.zeros(len(X), dtype=int)
outer_thresholds = []

outer_cv = StratifiedGroupKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=42)

for outer_fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y, groups)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]

    inner_probability = binary_oof(X_train, y_train, groups_train, INNER_SPLITS)
    threshold = choose_threshold(y_train, inner_probability)
    valid_probability = np.zeros(len(valid_idx))

    for seed in SEEDS:
        model = create_model(10_000 + outer_fold * 100 + seed)
        model.fit(X_train, y_train, cat_features=cat_cols)
        valid_probability += model.predict_proba(X_valid)[:, 1] / len(SEEDS)

    outer_probability[valid_idx] = valid_probability
    outer_prediction[valid_idx] = (valid_probability >= threshold).astype(int)
    outer_thresholds.append(threshold)


top_k = min(TOP_K, len(y))
top_idx = np.argsort(outer_probability)[::-1][:top_k]

metrics = pd.Series({
    "threshold_mean": np.mean(outer_thresholds),
    "threshold_median": np.median(outer_thresholds),
    "average_precision": average_precision_score(y, outer_probability),
    "precision": precision_score(y, outer_prediction, zero_division=0),
    "recall": recall_score(y, outer_prediction, zero_division=0),
    "f0.5": fbeta_score(y, outer_prediction, beta=0.5, zero_division=0),
    f"precision_at_{top_k}": y.iloc[top_idx].mean(),
    f"recall_at_{top_k}": y.iloc[top_idx].sum() / y.sum(),
    "predicted_hnwi": outer_prediction.sum()
})

display(metrics.round(4))
display(pd.Series(outer_thresholds, name="threshold_by_outer_fold").round(4))

display(pd.DataFrame(
    confusion_matrix(y, outer_prediction, labels=[0, 1]),
    index=["TRUE_NOT_HNWI", "TRUE_HNWI"],
    columns=["PRED_NOT_HNWI", "PRED_HNWI"]
))

print(classification_report(
    y,
    outer_prediction,
    labels=[0, 1],
    target_names=["NOT_HNWI", "HNWI"],
    digits=3,
    zero_division=0
))


# Фінальна production-модель після чесного outer-CV оцінювання
production_oof = binary_oof(X, y, groups, OUTER_SPLITS)
production_threshold = choose_threshold(y, production_oof)

final_models = []

for seed in SEEDS:
    model = create_model(seed)
    model.fit(X, y, cat_features=cat_cols)
    final_models.append(model)

print("Production threshold:", round(production_threshold, 4))


def predict_hnwi(new_data):
    X_new = new_data.reindex(columns=X.columns).copy().replace([np.inf, -np.inf], np.nan)

    for col in cat_cols:
        X_new[col] = X_new[col].astype("string").fillna("Missing").astype(str)

    score = np.mean([model.predict_proba(X_new)[:, 1] for model in final_models], axis=0)

    return pd.DataFrame({
        "hnwi_score": score,
        "pred_hnwi": (score >= production_threshold).astype(int)
    }, index=new_data.index)