import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, confusion_matrix, classification_report

MIN_RECALL = 0.35
TOP_K = 50
SEEDS = (42, 52, 62)

for col in cat_cols: X[col] = X[col].astype("string").fillna("Missing").astype(str)

def create_model(seed): return CatBoostClassifier(loss_function="Logloss", iterations=700, depth=4, learning_rate=0.03, l2_leaf_reg=10, random_strength=1, auto_class_weights="SqrtBalanced", random_seed=seed, verbose=False, allow_writing_files=False)

def hnwi_oof(X, y, groups, seeds=SEEDS):
    p_affluent = np.zeros(len(X))
    p_hnwi_conditional = np.zeros(len(X))

    for seed in seeds:
        fold_p_affluent = np.zeros(len(X))
        fold_p_hnwi = np.zeros(len(X))
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
            y_train = y.iloc[train_idx]

            affluent_model = create_model(seed + fold)
            affluent_model.fit(X.iloc[train_idx], (y_train >= 1).astype(int), cat_features=cat_cols)
            fold_p_affluent[valid_idx] = affluent_model.predict_proba(X.iloc[valid_idx])[:, 1]

            affluent_train_idx = train_idx[y_train.to_numpy() >= 1]
            hnwi_model = create_model(seed + 100 + fold)
            hnwi_model.fit(X.iloc[affluent_train_idx], (y.iloc[affluent_train_idx] == 2).astype(int), cat_features=cat_cols)
            fold_p_hnwi[valid_idx] = hnwi_model.predict_proba(X.iloc[valid_idx])[:, 1]

        p_affluent += fold_p_affluent / len(seeds)
        p_hnwi_conditional += fold_p_hnwi / len(seeds)

    return p_affluent, p_hnwi_conditional, p_affluent * p_hnwi_conditional

def choose_gate_thresholds(y_true, p_affluent, p_hnwi_conditional, min_recall=MIN_RECALL):
    affluent_grid = np.unique(np.r_[0, np.quantile(p_affluent, np.linspace(0.05, 0.99, 80)), 1])
    hnwi_grid = np.unique(np.r_[0, np.quantile(p_hnwi_conditional, np.linspace(0.05, 0.99, 80)), 1])
    best = None

    for affluent_threshold in affluent_grid:
        for hnwi_threshold in hnwi_grid:
            prediction = ((p_affluent >= affluent_threshold) & (p_hnwi_conditional >= hnwi_threshold)).astype(int)
            recall = recall_score(y_true, prediction, zero_division=0)

            if recall < min_recall: continue

            precision = precision_score(y_true, prediction, zero_division=0)
            f05 = fbeta_score(y_true, prediction, beta=0.5, zero_division=0)
            key = (precision, f05, recall, -prediction.sum())

            if best is None or key > best[0]: best = (key, affluent_threshold, hnwi_threshold)

    return best[1], best[2]

y_hnwi = (y.to_numpy() == 2).astype(int)
oof_p_affluent, oof_p_hnwi_conditional, oof_hnwi_score = hnwi_oof(X, y, groups)

affluent_threshold, hnwi_threshold = choose_gate_thresholds(y_hnwi, oof_p_affluent, oof_p_hnwi_conditional)
oof_prediction = ((oof_p_affluent >= affluent_threshold) & (oof_p_hnwi_conditional >= hnwi_threshold)).astype(int)

top_k = min(TOP_K, len(y_hnwi))
top_idx = np.argsort(oof_hnwi_score)[::-1][:top_k]

metrics = pd.Series({
    "affluent_threshold": affluent_threshold,
    "hnwi_conditional_threshold": hnwi_threshold,
    "average_precision": average_precision_score(y_hnwi, oof_hnwi_score),
    "precision": precision_score(y_hnwi, oof_prediction, zero_division=0),
    "recall": recall_score(y_hnwi, oof_prediction, zero_division=0),
    "f0.5": fbeta_score(y_hnwi, oof_prediction, beta=0.5, zero_division=0),
    f"precision_at_{top_k}": y_hnwi[top_idx].mean(),
    f"recall_at_{top_k}": y_hnwi[top_idx].sum() / y_hnwi.sum(),
    "predicted_hnwi": oof_prediction.sum()
})

display(metrics.round(4))
display(pd.DataFrame(confusion_matrix(y_hnwi, oof_prediction), index=["TRUE_NOT_HNWI", "TRUE_HNWI"], columns=["PRED_NOT_HNWI", "PRED_HNWI"]))
print(classification_report(y_hnwi, oof_prediction, target_names=["NOT_HNWI", "HNWI"], digits=3, zero_division=0))

affluent_model = create_model(1000)
hnwi_model = create_model(2000)
affluent_model.fit(X, (y >= 1).astype(int), cat_features=cat_cols)
affluent_mask = y >= 1
hnwi_model.fit(X.loc[affluent_mask], (y.loc[affluent_mask] == 2).astype(int), cat_features=cat_cols)

def predict_hnwi(X_new):
    X_new = X_new.copy()
    for col in cat_cols: X_new[col] = X_new[col].astype("string").fillna("Missing").astype(str)
    p_affluent = affluent_model.predict_proba(X_new)[:, 1]
    p_hnwi_conditional = hnwi_model.predict_proba(X_new)[:, 1]
    hnwi_score = p_affluent * p_hnwi_conditional
    prediction = ((p_affluent >= affluent_threshold) & (p_hnwi_conditional >= hnwi_threshold)).astype(int)
    return pd.DataFrame({"p_affluent": p_affluent, "p_hnwi_given_affluent": p_hnwi_conditional, "hnwi_score": hnwi_score, "prediction": np.where(prediction == 1, "HNWI", "NOT_HNWI")}, index=X_new.index)