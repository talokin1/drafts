import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score, roc_auc_score, confusion_matrix

SEED = 42
TEST_SIZE = 0.20
TARGET_PRECISION = 0.75
MIN_SELECTED = 5
PREM_WEIGHT = 2.0
HNWI_WEIGHT = 4.0

df = client_df.copy()
df["TARGET"] = (df["SEGMENT"] == "HNWI").astype(int)

LEAKAGE_COLS = ["SEGMENT", "TARGET", "MOBILEPHONE", "group_id"]
FEATURE_COLS = [c for c in df.columns if c not in LEAKAGE_COLS and df[c].nunique(dropna=False) > 1]

X = df[FEATURE_COLS].copy().replace([np.inf, -np.inf], np.nan)
y = df["TARGET"].copy()
segments = df["SEGMENT"].copy()

cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)

for col in bool_cols:
    X[col] = X[col].astype(int)

X_train, X_test, y_train, y_test, segment_train, segment_test = train_test_split(X, y, segments, test_size=TEST_SIZE, stratify=y, random_state=SEED)

def get_weights(segment):
    return segment.map({"MASS": 1.0, "PREM": PREM_WEIGHT, "HNWI": HNWI_WEIGHT}).astype(float)

def create_model(seed=SEED):
    return CatBoostClassifier(loss_function="Logloss", eval_metric="PRAUC", iterations=500, depth=3, learning_rate=0.03, l2_leaf_reg=10, random_strength=1.5, min_data_in_leaf=15, bootstrap_type="Bayesian", random_seed=seed, verbose=False, allow_writing_files=False)

def get_oof_predictions(X, y, segments):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    prediction_sum = np.zeros(len(X))
    prediction_count = np.zeros(len(X))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        model = create_model(SEED + fold)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_cols, sample_weight=get_weights(segments.iloc[train_idx]))
        prediction_sum[valid_idx] += model.predict_proba(X.iloc[valid_idx])[:, 1]
        prediction_count[valid_idx] += 1

    return prediction_sum / prediction_count

def find_threshold(y_true, probabilities, target_precision=TARGET_PRECISION, min_selected=MIN_SELECTED):
    results = []

    for threshold in np.unique(probabilities):
        predictions = (probabilities >= threshold).astype(int)
        selected = predictions.sum()

        if selected < min_selected:
            continue

        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f05 = fbeta_score(y_true, predictions, beta=0.5, zero_division=0)
        results.append([threshold, precision, recall, f05, selected])

    results = pd.DataFrame(results, columns=["threshold", "precision", "recall", "f0.5", "selected"])
    valid = results[results["precision"] >= target_precision]

    if len(valid):
        best = valid.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        best = results.sort_values(["f0.5", "precision", "recall"], ascending=[False, False, False]).iloc[0]

    return best, results

def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return pd.Series({
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f0.5": fbeta_score(y_true, predictions, beta=0.5, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "selected": predictions.sum(),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })

def precision_at_k(y_true, probabilities, k):
    k = min(k, len(y_true))
    top_idx = np.argsort(probabilities)[::-1][:k]
    return np.asarray(y_true)[top_idx].mean()

oof_probabilities = get_oof_predictions(X_train.reset_index(drop=True), y_train.reset_index(drop=True), segment_train.reset_index(drop=True))
best_threshold, threshold_table = find_threshold(y_train.reset_index(drop=True), oof_probabilities)
threshold = float(best_threshold["threshold"])

final_model = create_model()
final_model.fit(X_train, y_train, cat_features=cat_cols, sample_weight=get_weights(segment_train))

test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)

metrics = pd.DataFrame({
    "OOF_TRAIN": evaluate(y_train, oof_probabilities, threshold),
    "TEST": evaluate(y_test, test_probabilities, threshold)
}).T

ranking_metrics = pd.DataFrame({
    "K": [10, 20, 50],
    "Precision@K": [precision_at_k(y_test, test_probabilities, k) for k in [10, 20, 50]],
    "Lift@K": [precision_at_k(y_test, test_probabilities, k) / y_test.mean() for k in [10, 20, 50]]
})

test_result = df.loc[X_test.index, ["MOBILEPHONE", "SEGMENT"]].copy()
test_result["HNWI_SCORE"] = test_probabilities
test_result["HNWI_PREDICTION"] = test_predictions
test_result = test_result.sort_values("HNWI_SCORE", ascending=False)

false_positives = test_result[(test_result["SEGMENT"] != "HNWI") & (test_result["HNWI_PREDICTION"] == 1)]
false_negatives = test_result[(test_result["SEGMENT"] == "HNWI") & (test_result["HNWI_PREDICTION"] == 0)]

feature_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": final_model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("Обраний threshold:")
print(best_threshold)
print("\nМетрики:")
print(metrics.round(4))
print("\nRanking-метрики:")
print(ranking_metrics.round(4))
print("\nFalse positives за сегментами:")
print(false_positives["SEGMENT"].value_counts())
print("\nTop features:")
print(feature_importance.head(15))