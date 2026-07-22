import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, confusion_matrix, classification_report

MIN_RECALL = 0.08
MIN_PREDICTED = 8
SEEDS = (42, 52, 62)

for col in cat_cols: X[col] = X[col].astype("string").fillna("Missing").astype(str)

def create_model(seed, class_weights=None, auto_balance=False):
    params = dict(loss_function="Logloss", iterations=700, depth=4, learning_rate=0.03, l2_leaf_reg=12, random_strength=1, random_seed=seed, verbose=False, allow_writing_files=False)
    if auto_balance: params["auto_class_weights"] = "SqrtBalanced"
    if class_weights is not None: params["class_weights"] = class_weights
    return CatBoostClassifier(**params)

def hnwi_precision_oof(X, y, groups, seeds=SEEDS):
    p_affluent = np.zeros(len(X))
    p_hnwi_conditional = np.zeros(len(X))
    p_hnwi_verifier = np.zeros(len(X))

    for seed in seeds:
        fold_p_affluent = np.zeros(len(X))
        fold_p_conditional = np.zeros(len(X))
        fold_p_verifier = np.zeros(len(X))
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
            y_train = y.iloc[train_idx]

            affluent_model = create_model(seed + fold, auto_balance=True)
            affluent_model.fit(X.iloc[train_idx], (y_train >= 1).astype(int), cat_features=cat_cols)
            fold_p_affluent[valid_idx] = affluent_model.predict_proba(X.iloc[valid_idx])[:, 1]

            affluent_train_idx = train_idx[y_train.to_numpy() >= 1]
            hnwi_model = create_model(seed + 100 + fold, class_weights=[1.0, 1.0])
            hnwi_model.fit(X.iloc[affluent_train_idx], (y.iloc[affluent_train_idx] == 2).astype(int), cat_features=cat_cols)
            fold_p_conditional[valid_idx] = hnwi_model.predict_proba(X.iloc[valid_idx])[:, 1]

            verifier_model = create_model(seed + 200 + fold, class_weights=[1.0, 2.0])
            verifier_weights = np.where(y_train.to_numpy() == 1, 2.0, 1.0)
            verifier_model.fit(X.iloc[train_idx], (y_train == 2).astype(int), cat_features=cat_cols, sample_weight=verifier_weights)
            fold_p_verifier[valid_idx] = verifier_model.predict_proba(X.iloc[valid_idx])[:, 1]

        p_affluent += fold_p_affluent / len(seeds)
        p_hnwi_conditional += fold_p_conditional / len(seeds)
        p_hnwi_verifier += fold_p_verifier / len(seeds)

    hnwi_score = p_affluent * p_hnwi_conditional * p_hnwi_verifier
    return p_affluent, p_hnwi_conditional, p_hnwi_verifier, hnwi_score

def precision_lower_bound(tp, predicted, z=1.64):
    if predicted == 0: return 0.0
    p = tp / predicted
    denominator = 1 + z ** 2 / predicted
    centre = p + z ** 2 / (2 * predicted)
    margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * predicted)) / predicted)
    return (centre - margin) / denominator

def choose_precision_threshold(y_true, score, min_recall=MIN_RECALL, min_predicted=MIN_PREDICTED):
    grid = np.unique(np.r_[0, np.quantile(score, np.linspace(0.50, 0.999, 300)), 1])
    best = None

    for threshold in grid:
        prediction = (score >= threshold).astype(int)
        predicted = prediction.sum()
        tp = ((prediction == 1) & (y_true == 1)).sum()

        if predicted < min_predicted: continue

        recall = recall_score(y_true, prediction, zero_division=0)
        if recall < min_recall: continue

        precision = precision_score(y_true, prediction, zero_division=0)
        f025 = fbeta_score(y_true, prediction, beta=0.25, zero_division=0)
        precision_lcb = precision_lower_bound(tp, predicted)
        key = (precision_lcb, precision, f025, recall, -predicted)

        if best is None or key > best[0]: best = (key, threshold)

    if best is None: raise ValueError("Не знайдено threshold, який задовольняє MIN_RECALL і MIN_PREDICTED")

    return best[1]

y_hnwi = (y.to_numpy() == 2).astype(int)

oof_p_affluent, oof_p_hnwi_conditional, oof_p_hnwi_verifier, oof_hnwi_score = hnwi_precision_oof(X, y, groups)

hnwi_threshold = choose_precision_threshold(y_hnwi, oof_hnwi_score)
oof_prediction = (oof_hnwi_score >= hnwi_threshold).astype(int)

precision = precision_score(y_hnwi, oof_prediction, zero_division=0)
recall = recall_score(y_hnwi, oof_prediction, zero_division=0)

metrics = pd.Series({
    "hnwi_threshold": hnwi_threshold,
    "average_precision": average_precision_score(y_hnwi, oof_hnwi_score),
    "precision": precision,
    "recall": recall,
    "f0.25": fbeta_score(y_hnwi, oof_prediction, beta=0.25, zero_division=0),
    "predicted_hnwi": oof_prediction.sum(),
    "true_positive": ((oof_prediction == 1) & (y_hnwi == 1)).sum(),
    "false_positive": ((oof_prediction == 1) & (y_hnwi == 0)).sum(),
    "base_rate": y_hnwi.mean(),
    "lift_vs_random": precision / y_hnwi.mean()
})

display(metrics.round(4))
display(pd.DataFrame(confusion_matrix(y_hnwi, oof_prediction), index=["TRUE_NOT_HNWI", "TRUE_HNWI"], columns=["PRED_NOT_HNWI", "PRED_HNWI"]))
print(classification_report(y_hnwi, oof_prediction, target_names=["NOT_HNWI", "HNWI"], digits=3, zero_division=0))