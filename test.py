import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, confusion_matrix, classification_report

SEED = 42
SEEDS = (42, 52, 62)
MIN_RECALL = 0.15
MIN_PREDICTED = 5
TOP_K = 50
TARGET_MAP = {"MASS": 0, "PREM": 1, "HNWI": 2}

data = client_df.reset_index(drop=True).copy()
groups = data["group_id"].astype(str)
y = data["SEGMENT"].map(TARGET_MAP).astype(int)
X = data.drop(columns=["MOBILEPHONE", "group_id", "SEGMENT", "CONTRAGENTID"], errors="ignore").replace([np.inf, -np.inf], np.nan)

cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
for col in cat_cols: X[col] = X[col].astype("string").fillna("Missing").astype(str)

split_test = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
train_valid_idx, test_idx = next(split_test.split(X, y, groups))

X_train_valid = X.iloc[train_valid_idx].reset_index(drop=True)
y_train_valid = y.iloc[train_valid_idx].reset_index(drop=True)
groups_train_valid = groups.iloc[train_valid_idx].reset_index(drop=True)

split_valid = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
train_idx, valid_idx = next(split_valid.split(X_train_valid, y_train_valid, groups_train_valid))

X_train = X_train_valid.iloc[train_idx].reset_index(drop=True)
y_train = y_train_valid.iloc[train_idx].reset_index(drop=True)
groups_train = groups_train_valid.iloc[train_idx].reset_index(drop=True)

X_valid = X_train_valid.iloc[valid_idx].reset_index(drop=True)
y_valid = y_train_valid.iloc[valid_idx].reset_index(drop=True)

X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

print("TRAIN:", len(y_train), y_train.value_counts().sort_index().to_dict())
print("VALID:", len(y_valid), y_valid.value_counts().sort_index().to_dict())
print("TEST:", len(y_test), y_test.value_counts().sort_index().to_dict())


def create_model(seed):
    return CatBoostClassifier(loss_function="Logloss", iterations=700, depth=4, learning_rate=0.03, l2_leaf_reg=10, random_strength=1, auto_class_weights="SqrtBalanced", random_seed=seed, verbose=False, allow_writing_files=False)

def two_stage_oof(X, y, groups, seeds=SEEDS):
    p_affluent = np.zeros(len(X))
    p_hnwi_conditional = np.zeros(len(X))

    for seed in seeds:
        seed_p_affluent = np.zeros(len(X))
        seed_p_hnwi = np.zeros(len(X))
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)

        for fold, (fit_idx, oof_idx) in enumerate(cv.split(X, y, groups)):
            y_fit = y.iloc[fit_idx]

            affluent_model = create_model(seed + fold)
            affluent_model.fit(X.iloc[fit_idx], (y_fit >= 1).astype(int), cat_features=cat_cols)
            seed_p_affluent[oof_idx] = affluent_model.predict_proba(X.iloc[oof_idx])[:, 1]

            affluent_fit_idx = fit_idx[y_fit.to_numpy() >= 1]
            hnwi_target = (y.iloc[affluent_fit_idx] == 2).astype(int)

            if hnwi_target.nunique() < 2: raise ValueError(f"Fold {fold}: Stage 2 містить лише один клас")

            hnwi_model = create_model(seed + 100 + fold)
            hnwi_model.fit(X.iloc[affluent_fit_idx], hnwi_target, cat_features=cat_cols)
            seed_p_hnwi[oof_idx] = hnwi_model.predict_proba(X.iloc[oof_idx])[:, 1]

        p_affluent += seed_p_affluent / len(seeds)
        p_hnwi_conditional += seed_p_hnwi / len(seeds)

    return p_affluent, p_hnwi_conditional, p_affluent * p_hnwi_conditional

def choose_gate_thresholds(y_true, p_affluent, p_hnwi_conditional, min_recall=MIN_RECALL, min_predicted=MIN_PREDICTED):
    affluent_grid = np.unique(np.r_[0, np.quantile(p_affluent, np.linspace(0.05, 0.995, 120)), 1])
    hnwi_grid = np.unique(np.r_[0, np.quantile(p_hnwi_conditional, np.linspace(0.05, 0.995, 120)), 1])
    best = None

    for affluent_threshold in affluent_grid:
        for hnwi_threshold in hnwi_grid:
            prediction = ((p_affluent >= affluent_threshold) & (p_hnwi_conditional >= hnwi_threshold)).astype(int)
            predicted_count = prediction.sum()

            if predicted_count < min_predicted: continue

            recall = recall_score(y_true, prediction, zero_division=0)
            if recall < min_recall: continue

            precision = precision_score(y_true, prediction, zero_division=0)
            f05 = fbeta_score(y_true, prediction, beta=0.5, zero_division=0)
            key = (precision, f05, recall, -predicted_count)

            if best is None or key > best[0]: best = (key, affluent_threshold, hnwi_threshold)

    if best is None: raise ValueError("Не знайдено thresholds, які задовольняють обмеження")

    return best[1], best[2]

y_train_hnwi = (y_train.to_numpy() == 2).astype(int)

train_oof_p_affluent, train_oof_p_hnwi, train_oof_score = two_stage_oof(X_train, y_train, groups_train)

affluent_threshold, hnwi_threshold = choose_gate_thresholds(y_train_hnwi, train_oof_p_affluent, train_oof_p_hnwi)

print("Affluent threshold:", round(affluent_threshold, 4))
print("HNWI conditional threshold:", round(hnwi_threshold, 4))

def fit_two_stage_ensemble(X, y, seeds=SEEDS):
    affluent_models = []
    hnwi_models = []
    affluent_mask = y >= 1

    for seed in seeds:
        affluent_model = create_model(seed)
        affluent_model.fit(X, (y >= 1).astype(int), cat_features=cat_cols)
        affluent_models.append(affluent_model)

        hnwi_model = create_model(seed + 100)
        hnwi_model.fit(X.loc[affluent_mask], (y.loc[affluent_mask] == 2).astype(int), cat_features=cat_cols)
        hnwi_models.append(hnwi_model)

    return affluent_models, hnwi_models

affluent_models, hnwi_models = fit_two_stage_ensemble(X_train, y_train)


def predict_probabilities(X_new):
    p_affluent = np.mean([model.predict_proba(X_new)[:, 1] for model in affluent_models], axis=0)
    p_hnwi_conditional = np.mean([model.predict_proba(X_new)[:, 1] for model in hnwi_models], axis=0)
    return p_affluent, p_hnwi_conditional, p_affluent * p_hnwi_conditional


def evaluate_split(name, X_eval, y_eval):
    y_hnwi = (y_eval.to_numpy() == 2).astype(int)
    p_affluent, p_hnwi_conditional, hnwi_score = predict_probabilities(X_eval)
    prediction = ((p_affluent >= affluent_threshold) & (p_hnwi_conditional >= hnwi_threshold)).astype(int)

    top_k = min(TOP_K, len(y_eval))
    top_idx = np.argsort(hnwi_score)[::-1][:top_k]

    metrics = pd.Series({
        "average_precision": average_precision_score(y_hnwi, hnwi_score),
        "precision": precision_score(y_hnwi, prediction, zero_division=0),
        "recall": recall_score(y_hnwi, prediction, zero_division=0),
        "f0.5": fbeta_score(y_hnwi, prediction, beta=0.5, zero_division=0),
        f"precision_at_{top_k}": y_hnwi[top_idx].mean(),
        f"recall_at_{top_k}": y_hnwi[top_idx].sum() / max(y_hnwi.sum(), 1),
        "predicted_hnwi": prediction.sum(),
        "true_positive": ((prediction == 1) & (y_hnwi == 1)).sum(),
        "false_positive": ((prediction == 1) & (y_hnwi == 0)).sum(),
        "true_hnwi": y_hnwi.sum()
    })

    print(f"\n{name}")
    display(metrics.round(4))
    display(pd.DataFrame(confusion_matrix(y_hnwi, prediction), index=["TRUE_NOT_HNWI", "TRUE_HNWI"], columns=["PRED_NOT_HNWI", "PRED_HNWI"]))
    print(classification_report(y_hnwi, prediction, target_names=["NOT_HNWI", "HNWI"], digits=3, zero_division=0))

    return pd.DataFrame({"p_affluent": p_affluent, "p_hnwi_given_affluent": p_hnwi_conditional, "hnwi_score": hnwi_score, "prediction": np.where(prediction == 1, "HNWI", "NOT_HNWI")})

valid_predictions = evaluate_split("VALIDATION", X_valid, y_valid)
test_predictions = evaluate_split("TEST", X_test, y_test)