import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, confusion_matrix, classification_report

MIN_RECALL = 0.15
TOP_K = 50
SEEDS = (42, 52, 62)

for col in cat_cols: X[col] = X[col].astype("string").fillna("Missing").astype(str)

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

def binary_oof(X, y, groups, seeds=SEEDS):
    oof_probability = np.zeros(len(X))

    for seed in seeds:
        seed_probability = np.zeros(len(X))
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
            model = create_model(seed + fold)
            model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_cols)
            seed_probability[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

        oof_probability += seed_probability / len(seeds)

    return oof_probability

def choose_threshold(y_true, probability, min_recall=MIN_RECALL):
    thresholds = np.unique(np.r_[0, np.quantile(probability, np.linspace(0.05, 0.995, 200)), 1])
    best = None

    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)
        recall = recall_score(y_true, prediction, zero_division=0)

        if recall < min_recall: continue

        precision = precision_score(y_true, prediction, zero_division=0)
        f05 = fbeta_score(y_true, prediction, beta=0.5, zero_division=0)
        key = (precision, f05, recall, -prediction.sum())

        if best is None or key > best[0]: best = (key, threshold)

    if best is None: raise ValueError("Не знайдено threshold із заданим minimum recall")

    return best[1]


oof_hnwi_score = binary_oof(X, y, groups)
hnwi_threshold = choose_threshold(y, oof_hnwi_score)
oof_prediction = (oof_hnwi_score >= hnwi_threshold).astype(int)

top_k = min(TOP_K, len(y))
top_idx = np.argsort(oof_hnwi_score)[::-1][:top_k]

metrics = pd.Series({
    "hnwi_threshold": hnwi_threshold,
    "average_precision": average_precision_score(y, oof_hnwi_score),
    "precision": precision_score(y, oof_prediction, zero_division=0),
    "recall": recall_score(y, oof_prediction, zero_division=0),
    "f0.5": fbeta_score(y, oof_prediction, beta=0.5, zero_division=0),
    f"precision_at_{top_k}": y.iloc[top_idx].mean(),
    f"recall_at_{top_k}": y.iloc[top_idx].sum() / y.sum(),
    "predicted_hnwi": oof_prediction.sum()
})

display(metrics.round(4))
display(pd.DataFrame(confusion_matrix(y, oof_prediction), index=["TRUE_MASS", "TRUE_HNWI"], columns=["PRED_MASS", "PRED_HNWI"]))
print(classification_report(y, oof_prediction, target_names=["MASS", "HNWI"], digits=3, zero_division=0))