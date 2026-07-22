import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, confusion_matrix, classification_report

MIN_RECALL = 0.15
MIN_PREDICTED = 10
HARD_MASS_QUANTILE = 0.85
SEEDS = (42, 52)
N_OUTER = 4
N_INNER = 3

X = X.reset_index(drop=True).copy()
y = pd.Series(np.asarray(y), name="target")
groups = pd.Series(np.asarray(groups), name="group")
y_hnwi = (y == 2).astype(int)

for col in cat_cols: X[col] = X[col].astype("string").fillna("Missing").astype(str)

def create_model(seed): return CatBoostClassifier(loss_function="Logloss", eval_metric="PRAUC", iterations=700, depth=4, learning_rate=0.03, l2_leaf_reg=10, random_strength=1, random_seed=seed, verbose=False, allow_writing_files=False)

def make_weights(y_binary, y_ordinal, reranker=False):
    positive_weight = np.sqrt((y_binary == 0).sum() / max((y_binary == 1).sum(), 1))
    negative_weight = np.where(y_ordinal.to_numpy() == 1, 2.0, 1.5 if reranker else 1.0)
    return np.where(y_binary.to_numpy() == 1, positive_weight, negative_weight)

def repeated_oof(X_part, y_binary, y_ordinal, group_part, train_mask=None, n_splits=3, seed_offset=0):
    probabilities = np.zeros((len(X_part), len(SEEDS)))
    allowed = np.ones(len(X_part), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)

    for seed_col, seed in enumerate(SEEDS):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + seed_offset)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_part, y_binary, group_part)):
            fit_idx = train_idx[allowed[train_idx]]

            if y_binary.iloc[fit_idx].nunique() < 2: probabilities[valid_idx, seed_col] = y_binary.iloc[fit_idx].mean(); continue

            model = create_model(seed + seed_offset + fold)
            model.fit(X_part.iloc[fit_idx], y_binary.iloc[fit_idx], cat_features=cat_cols, sample_weight=make_weights(y_binary.iloc[fit_idx], y_ordinal.iloc[fit_idx], train_mask is not None))
            probabilities[valid_idx, seed_col] = model.predict_proba(X_part.iloc[valid_idx])[:, 1]

    return probabilities

def make_hard_mask(y_ordinal, base_score):
    mass_mask = y_ordinal.to_numpy() == 0
    mass_threshold = np.quantile(base_score[mass_mask], HARD_MASS_QUANTILE)
    return (y_ordinal.to_numpy() >= 1) | (mass_mask & (base_score >= mass_threshold))

def wilson_lower(y_true, prediction, z=1.96):
    n = prediction.sum()

    if n == 0: return 0.0

    p = y_true[prediction == 1].mean()
    return (p + z ** 2 / (2 * n) - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / (1 + z ** 2 / n)

def choose_thresholds(y_true, base_score, rerank_score, uncertainty, min_recall=MIN_RECALL, min_predicted=MIN_PREDICTED):
    base_grid = np.unique(np.r_[0, np.quantile(base_score, np.linspace(0.05, 0.995, 45)), 1])
    rerank_grid = np.unique(np.r_[0, np.quantile(rerank_score, np.linspace(0.05, 0.995, 45)), 1])
    uncertainty_grid = np.unique(np.r_[np.quantile(uncertainty, np.linspace(0.60, 1.00, 5)), np.inf])
    best = None

    for base_threshold in base_grid:
        for rerank_threshold in rerank_grid:
            gate = (base_score >= base_threshold) & (rerank_score >= rerank_threshold)

            for max_uncertainty in uncertainty_grid:
                prediction = (gate & (uncertainty <= max_uncertainty)).astype(int)

                if prediction.sum() < min_predicted: continue

                recall = recall_score(y_true, prediction, zero_division=0)

                if recall < min_recall: continue

                precision = precision_score(y_true, prediction, zero_division=0)
                f05 = fbeta_score(y_true, prediction, beta=0.5, zero_division=0)
                key = (wilson_lower(y_true, prediction), precision, f05, recall, -prediction.sum())

                if best is None or key > best[0]: best = (key, base_threshold, rerank_threshold, max_uncertainty)

    if best is None and (min_recall > 0 or min_predicted > 1): return choose_thresholds(y_true, base_score, rerank_score, uncertainty, 0, 1)

    return best[1], best[2], best[3]

def fit_models(X_part, y_binary, y_ordinal, train_mask=None, seed_offset=0):
    fit_idx = np.arange(len(X_part)) if train_mask is None else np.flatnonzero(train_mask)
    models = []

    for seed in SEEDS:
        model = create_model(seed + seed_offset)
        model.fit(X_part.iloc[fit_idx], y_binary.iloc[fit_idx], cat_features=cat_cols, sample_weight=make_weights(y_binary.iloc[fit_idx], y_ordinal.iloc[fit_idx], train_mask is not None))
        models.append(model)

    return models

def ensemble_predict(models, X_part):
    probabilities = np.column_stack([model.predict_proba(X_part)[:, 1] for model in models])
    return probabilities.mean(axis=1), probabilities.std(axis=1)

def nested_hnwi_oof(X, y, y_binary, groups):
    oof_base = np.zeros(len(X))
    oof_rerank = np.zeros(len(X))
    oof_uncertainty = np.zeros(len(X))
    oof_prediction = np.zeros(len(X), dtype=int)
    fold_thresholds = []
    outer_cv = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y_binary, groups)):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_binary_train = y_binary.iloc[train_idx].reset_index(drop=True)
        groups_train = groups.iloc[train_idx].reset_index(drop=True)

        inner_base = repeated_oof(X_train, y_binary_train, y_train, groups_train, n_splits=N_INNER)
        inner_base_mean = inner_base.mean(axis=1)
        inner_base_std = inner_base.std(axis=1)

        hard_mask = make_hard_mask(y_train, inner_base_mean)

        inner_rerank = repeated_oof(X_train, y_binary_train, y_train, groups_train, train_mask=hard_mask, n_splits=N_INNER, seed_offset=1000)
        inner_rerank_mean = inner_rerank.mean(axis=1)
        inner_rerank_std = inner_rerank.std(axis=1)

        base_threshold, rerank_threshold, max_uncertainty = choose_thresholds(y_binary_train.to_numpy(), inner_base_mean, inner_rerank_mean, np.maximum(inner_base_std, inner_rerank_std))

        base_models = fit_models(X_train, y_binary_train, y_train)
        rerank_models = fit_models(X_train, y_binary_train, y_train, train_mask=hard_mask, seed_offset=1000)

        base_mean, base_std = ensemble_predict(base_models, X_valid)
        rerank_mean, rerank_std = ensemble_predict(rerank_models, X_valid)
        uncertainty = np.maximum(base_std, rerank_std)
        prediction = ((base_mean >= base_threshold) & (rerank_mean >= rerank_threshold) & (uncertainty <= max_uncertainty)).astype(int)

        oof_base[valid_idx] = base_mean
        oof_rerank[valid_idx] = rerank_mean
        oof_uncertainty[valid_idx] = uncertainty
        oof_prediction[valid_idx] = prediction

        fold_thresholds.append({"fold": fold, "base_threshold": base_threshold, "rerank_threshold": rerank_threshold, "max_uncertainty": max_uncertainty, "hard_train_size": int(hard_mask.sum())})

    return oof_base, oof_rerank, oof_uncertainty, oof_prediction, pd.DataFrame(fold_thresholds)

oof_base, oof_rerank, oof_uncertainty, oof_prediction, fold_thresholds = nested_hnwi_oof(X, y, y_hnwi, groups)
oof_score = np.sqrt(oof_base * oof_rerank)

top_k = min(50, len(y_hnwi))
top_idx = np.argsort(oof_score)[::-1][:top_k]

metrics = pd.Series({
    "average_precision": average_precision_score(y_hnwi, oof_score),
    "precision": precision_score(y_hnwi, oof_prediction, zero_division=0),
    "recall": recall_score(y_hnwi, oof_prediction, zero_division=0),
    "f0.5": fbeta_score(y_hnwi, oof_prediction, beta=0.5, zero_division=0),
    f"precision_at_{top_k}": y_hnwi.iloc[top_idx].mean(),
    f"recall_at_{top_k}": y_hnwi.iloc[top_idx].sum() / y_hnwi.sum(),
    "predicted_hnwi": oof_prediction.sum()
})

display(metrics.round(4))
display(fold_thresholds.round(4))
display(pd.DataFrame(confusion_matrix(y_hnwi, oof_prediction), index=["TRUE_NOT_HNWI", "TRUE_HNWI"], columns=["PRED_NOT_HNWI", "PRED_HNWI"]))
print(classification_report(y_hnwi, oof_prediction, target_names=["NOT_HNWI", "HNWI"], digits=3, zero_division=0))