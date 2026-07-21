import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, precision_recall_curve

TARGET_NAMES = np.array(["MASS", "PREM", "HNWI"])

for col in cat_cols:
    X[col] = X[col].astype("string").fillna("Missing").astype(str)

cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

def create_model(seed):
    return CatBoostClassifier(loss_function="Logloss", iterations=700, depth=4, learning_rate=0.03, l2_leaf_reg=10, random_strength=1, auto_class_weights="SqrtBalanced", random_seed=seed, verbose=False, allow_writing_files=False)

def two_stage_oof(X, y, groups):
    p_affluent = np.zeros(len(X))
    p_hnwi_conditional = np.zeros(len(X))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
        y_train = y.iloc[train_idx]

        affluent_model = create_model(100 + fold)
        affluent_model.fit(X.iloc[train_idx], (y_train >= 1).astype(int), cat_features=cat_cols)
        p_affluent[valid_idx] = affluent_model.predict_proba(X.iloc[valid_idx])[:, 1]

        affluent_train_idx = train_idx[y_train.to_numpy() >= 1]
        hnwi_model = create_model(200 + fold)
        hnwi_model.fit(X.iloc[affluent_train_idx], (y.iloc[affluent_train_idx] == 2).astype(int), cat_features=cat_cols)
        p_hnwi_conditional[valid_idx] = hnwi_model.predict_proba(X.iloc[valid_idx])[:, 1]

    p_mass = 1 - p_affluent
    p_hnwi = p_affluent * p_hnwi_conditional
    p_prem = p_affluent * (1 - p_hnwi_conditional)

    return np.column_stack([p_mass, p_prem, p_hnwi]), p_affluent

oof_probabilities, oof_p_affluent = two_stage_oof(X, y, groups)

oof_p_mass = oof_probabilities[:, 0]
oof_p_prem = oof_probabilities[:, 1]
oof_p_hnwi = oof_probabilities[:, 2]



def best_threshold(y_true, score, beta=1):
    precision, recall, thresholds = precision_recall_curve(y_true, score)

    if len(thresholds) == 0:
        return 0.5

    f_beta = (1 + beta ** 2) * precision[:-1] * recall[:-1] / (beta ** 2 * precision[:-1] + recall[:-1] + 1e-12)
    return thresholds[np.argmax(f_beta)]


affluent_threshold = best_threshold((y >= 1).astype(int), oof_p_affluent, beta=1)
hnwi_threshold = best_threshold((y == 2).astype(int), oof_p_hnwi, beta=2)

print("AFFLUENT threshold:", round(affluent_threshold, 4))
print("HNWI threshold:", round(hnwi_threshold, 4))


oof_prediction = np.where(oof_p_hnwi >= hnwi_threshold, 2, np.where(oof_p_affluent >= affluent_threshold, 1, 0))

top_k = min(50, len(y))
top_idx = np.argsort(oof_p_hnwi)[::-1][:top_k]
hnwi_total = (y == 2).sum()

metrics = pd.Series({
    "macro_f1": f1_score(y, oof_prediction, average="macro"),
    "balanced_accuracy": balanced_accuracy_score(y, oof_prediction),
    "quadratic_kappa": cohen_kappa_score(y, oof_prediction, weights="quadratic"),
    "ordinal_mae": mean_absolute_error(y, oof_prediction),
    "HNWI_average_precision": average_precision_score((y == 2).astype(int), oof_p_hnwi),
    f"precision_at_{top_k}": np.mean(y.iloc[top_idx].to_numpy() == 2),
    f"recall_at_{top_k}": np.sum(y.iloc[top_idx].to_numpy() == 2) / hnwi_total,
    "HNWI_recall": np.mean(oof_prediction[y.to_numpy() == 2] == 2),
    "HNWI_to_MASS_rate": np.mean(oof_prediction[y.to_numpy() == 2] == 0)
})

display(metrics.round(4))

display(pd.DataFrame(confusion_matrix(y, oof_prediction, labels=[0, 1, 2]), index=["TRUE_MASS", "TRUE_PREM", "TRUE_HNWI"], columns=["PRED_MASS", "PRED_PREM", "PRED_HNWI"]))

print(classification_report(y, oof_prediction, labels=[0, 1, 2], target_names=TARGET_NAMES, digits=3, zero_division=0))

y_affluent = (y >= 1).astype(int)
affluent_mask = y >= 1
y_hnwi_affluent = (y.loc[affluent_mask] == 2).astype(int)

final_affluent_model = create_model(501)
final_affluent_model.fit(X, y_affluent, cat_features=cat_cols)

final_hnwi_model = create_model(502)
final_hnwi_model.fit(X.loc[affluent_mask], y_hnwi_affluent, cat_features=cat_cols)