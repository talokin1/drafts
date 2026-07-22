import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, confusion_matrix, classification_report

SEED = 42
MIN_RECALL = 0.15
TARGET_MAP = {"MASS": 0, "HNWI": 1}

data = client_df.reset_index(drop=True).copy()
groups = data["group_id"].astype(str)
y = data["SEGMENT"].map(TARGET_MAP).astype(int)
X = data.drop(columns=["MOBILEPHONE", "group_id", "SEGMENT", "CONTRAGENTID"], errors="ignore").replace([np.inf, -np.inf], np.nan)

cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
for col in cat_cols: X[col] = X[col].astype("string").fillna("Missing").astype(str)


outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_val_idx, test_idx = next(outer_cv.split(X, y, groups))

X_train_val = X.iloc[train_val_idx].reset_index(drop=True)
y_train_val = y.iloc[train_val_idx].reset_index(drop=True)
groups_train_val = groups.iloc[train_val_idx].reset_index(drop=True)

X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)
groups_test = groups.iloc[test_idx].reset_index(drop=True)

inner_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
train_idx, valid_idx = next(inner_split.split(X_train_val, y_train_val, groups_train_val))

X_train = X_train_val.iloc[train_idx].reset_index(drop=True)
y_train = y_train_val.iloc[train_idx].reset_index(drop=True)
groups_train = groups_train_val.iloc[train_idx].reset_index(drop=True)

X_valid = X_train_val.iloc[valid_idx].reset_index(drop=True)
y_valid = y_train_val.iloc[valid_idx].reset_index(drop=True)

print("TRAIN:", X_train.shape, y_train.value_counts().to_dict())
print("VALID:", X_valid.shape, y_valid.value_counts().to_dict())
print("TEST:", X_test.shape, y_test.value_counts().to_dict())


param_grid = [
    {"depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 10},
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10},
    {"depth": 4, "learning_rate": 0.02, "l2_leaf_reg": 20},
    {"depth": 5, "learning_rate": 0.02, "l2_leaf_reg": 20},
]

def create_model(params, seed):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=700,
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_strength=1,
        auto_class_weights="SqrtBalanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False
    )

def train_oof_score(X_train, y_train, groups_train, params):
    oof_score = np.zeros(len(X_train))
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)

    for fold, (fit_idx, oof_idx) in enumerate(cv.split(X_train, y_train, groups_train)):
        model = create_model(params, SEED + fold)
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx], cat_features=cat_cols)
        oof_score[oof_idx] = model.predict_proba(X_train.iloc[oof_idx])[:, 1]

    return oof_score

tuning_results = []

for params in param_grid:
    train_oof = train_oof_score(X_train, y_train, groups_train, params)
    tuning_results.append({
        **params,
        "train_oof_ap": average_precision_score(y_train, train_oof)
    })

tuning_results = pd.DataFrame(tuning_results).sort_values("train_oof_ap", ascending=False)
display(tuning_results)

best_params = tuning_results.iloc[0][["depth", "learning_rate", "l2_leaf_reg"]].to_dict()
best_params["depth"] = int(best_params["depth"])
best_params

model = create_model(best_params, SEED)
model.fit(X_train, y_train, cat_features=cat_cols)

valid_score = model.predict_proba(X_valid)[:, 1]

def choose_threshold(y_true, probability, min_recall=MIN_RECALL):
    thresholds = np.unique(np.r_[0, np.quantile(probability, np.linspace(0.05, 0.995, 300)), 1])
    best = None

    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)
        recall = recall_score(y_true, prediction, zero_division=0)

        if recall < min_recall:
            continue

        precision = precision_score(y_true, prediction, zero_division=0)
        f05 = fbeta_score(y_true, prediction, beta=0.5, zero_division=0)
        key = (precision, f05, recall, -prediction.sum())

        if best is None or key > best[0]:
            best = (key, threshold)

    if best is None:
        raise ValueError("Не знайдено threshold із заданим MIN_RECALL")

    return best[1]

hnwi_threshold = choose_threshold(y_valid, valid_score)
valid_prediction = (valid_score >= hnwi_threshold).astype(int)

print("Threshold:", hnwi_threshold)
print(classification_report(y_valid, valid_prediction, target_names=["MASS", "HNWI"], digits=3, zero_division=0))

test_score = model.predict_proba(X_test)[:, 1]
test_prediction = (test_score >= hnwi_threshold).astype(int)

test_metrics = pd.Series({
    "threshold": hnwi_threshold,
    "average_precision": average_precision_score(y_test, test_score),
    "precision": precision_score(y_test, test_prediction, zero_division=0),
    "recall": recall_score(y_test, test_prediction, zero_division=0),
    "f0.5": fbeta_score(y_test, test_prediction, beta=0.5, zero_division=0),
    "predicted_hnwi": test_prediction.sum(),
    "true_hnwi": y_test.sum(),
    "true_positive": ((test_prediction == 1) & (y_test.to_numpy() == 1)).sum(),
    "false_positive": ((test_prediction == 1) & (y_test.to_numpy() == 0)).sum()
})

display(test_metrics.round(4))

display(pd.DataFrame(
    confusion_matrix(y_test, test_prediction),
    index=["TRUE_MASS", "TRUE_HNWI"],
    columns=["PRED_MASS", "PRED_HNWI"]
))

print(classification_report(
    y_test,
    test_prediction,
    target_names=["MASS", "HNWI"],
    digits=3,
    zero_division=0
))

















false_positive_mask = (test_prediction == 1) & (y_test.to_numpy() == 0)

false_positives = (
    data.iloc[test_idx]
    .reset_index(drop=True)
    .loc[false_positive_mask]
    .assign(hnwi_score=test_score[false_positive_mask])
    .sort_values("hnwi_score", ascending=False)
)

display(false_positives)