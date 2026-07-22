import numpy as np
import pandas as pd

from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, precision_recall_curve


TARGET_MAP = {"MASS": 0, "PREM": 1, "HNWI": 2}
TARGET_NAMES = np.array(["MASS", "PREM", "HNWI"])

data = client_df.reset_index(drop=True).copy()
groups = data["group_id"].astype(str)
y = data["SEGMENT"].map(TARGET_MAP).astype(int)
X = data.drop(columns=["MOBILEPHONE", "group_id", "SEGMENT", "CONTRAGENTID"], errors="ignore").replace([np.inf, -np.inf], np.nan)

cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("string").fillna("Missing").astype(str)

def create_scorecard(X_train, y_train, max_bins=4, min_bin_size=0.10):
    cat_params = {col: {"cat_cutoff": 0.05} for col in cat_cols}
    binning = BinningProcess(variable_names=X_train.columns.tolist(), max_n_prebins=10, max_n_bins=max_bins, min_prebin_size=min_bin_size, min_bin_size=min_bin_size, binning_fit_params=cat_params, n_jobs=-1)
    estimator = LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=3000)
    model = Scorecard(binning_process=binning, estimator=estimator, scaling_method="min_max", scaling_method_params={"min": 0, "max": 100}, intercept_based=True, reverse_scorecard=True)
    return model.fit(X_train, y_train)





cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

p_affluent = np.zeros(len(X))
p_hnwi_conditional = np.zeros(len(X))

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
    affluent_model = create_scorecard(X.iloc[train_idx], (y.iloc[train_idx] >= 1).astype(int), max_bins=4, min_bin_size=0.10)
    p_affluent[valid_idx] = affluent_model.predict_proba(X.iloc[valid_idx])[:, 1]

    stage2_idx = train_idx[y.iloc[train_idx].to_numpy() >= 1]
    hnwi_model = create_scorecard(X.iloc[stage2_idx], (y.iloc[stage2_idx] == 2).astype(int), max_bins=3, min_bin_size=0.12)
    p_hnwi_conditional[valid_idx] = hnwi_model.predict_proba(X.iloc[valid_idx])[:, 1]

p_mass = 1 - p_affluent
p_hnwi = p_affluent * p_hnwi_conditional
p_prem = p_affluent * (1 - p_hnwi_conditional)
oof_probabilities = np.column_stack([p_mass, p_prem, p_hnwi])





def best_threshold(y_true, score, beta=1):
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f_beta = (1 + beta ** 2) * precision[:-1] * recall[:-1] / (beta ** 2 * precision[:-1] + recall[:-1] + 1e-12)
    return thresholds[np.nanargmax(f_beta)]

threshold_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=123)
oof_prediction = np.zeros(len(y), dtype=int)
fold_thresholds = []

for train_idx, valid_idx in threshold_cv.split(X, y, groups):
    affluent_threshold = best_threshold((y.iloc[train_idx] >= 1).astype(int), p_affluent[train_idx], beta=1)
    hnwi_threshold = best_threshold((y.iloc[train_idx] == 2).astype(int), p_hnwi[train_idx], beta=1)
    oof_prediction[valid_idx] = np.where(p_hnwi[valid_idx] >= hnwi_threshold, 2, np.where(p_affluent[valid_idx] >= affluent_threshold, 1, 0))
    fold_thresholds.append([affluent_threshold, hnwi_threshold])

pd.DataFrame(fold_thresholds, columns=["AFFLUENT_THRESHOLD", "HNWI_THRESHOLD"])







top_k = min(50, len(y))
top_idx = np.argsort(p_hnwi)[::-1][:top_k]
y_array = y.to_numpy()
hnwi_total = np.sum(y_array == 2)

metrics = pd.Series({
    "macro_f1": f1_score(y, oof_prediction, average="macro"),
    "balanced_accuracy": balanced_accuracy_score(y, oof_prediction),
    "quadratic_kappa": cohen_kappa_score(y, oof_prediction, weights="quadratic"),
    "ordinal_mae": mean_absolute_error(y, oof_prediction),
    "AFFLUENT_average_precision": average_precision_score((y >= 1).astype(int), p_affluent),
    "HNWI_conditional_AP": average_precision_score((y[y >= 1] == 2).astype(int), p_hnwi_conditional[y >= 1]),
    "HNWI_average_precision": average_precision_score((y == 2).astype(int), p_hnwi),
    f"precision_at_{top_k}": np.mean(y_array[top_idx] == 2),
    f"recall_at_{top_k}": np.sum(y_array[top_idx] == 2) / hnwi_total,
    "HNWI_recall": np.mean(oof_prediction[y_array == 2] == 2),
    "HNWI_to_MASS_rate": np.mean(oof_prediction[y_array == 2] == 0)
})

display(metrics.round(4))
display(pd.DataFrame(confusion_matrix(y, oof_prediction, labels=[0, 1, 2]), index=["TRUE_MASS", "TRUE_PREM", "TRUE_HNWI"], columns=["PRED_MASS", "PRED_PREM", "PRED_HNWI"]))
print(classification_report(y, oof_prediction, labels=[0, 1, 2], target_names=TARGET_NAMES, digits=3, zero_division=0))


affluent_target = (y >= 1).astype(int)
affluent_mask = y >= 1

final_affluent_model = create_scorecard(X, affluent_target, max_bins=4, min_bin_size=0.10)
final_hnwi_model = create_scorecard(X.loc[affluent_mask], (y.loc[affluent_mask] == 2).astype(int), max_bins=3, min_bin_size=0.12)

final_affluent_threshold = best_threshold(affluent_target, p_affluent, beta=1)
final_hnwi_threshold = best_threshold((y == 2).astype(int), p_hnwi, beta=1)

print("AFFLUENT threshold:", round(final_affluent_threshold, 4))
print("HNWI threshold:", round(final_hnwi_threshold, 4))