import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support
)


TARGET_COL = "SEGMENT"

# ID-колонки, які не треба давати в модель
ID_COLS = [
    "CONTRAGENTID",
    "CLIENT_ID",
    "MOBILEPHONE",
]

# Колонки, які можуть давати leakage.
# Якщо SEGMENT побудований з BALANCE, то BALANCE НЕ можна давати в модель.
LEAKAGE_COLS = [
    "SEGMENT",
    "BALANCE",
]

# Залишаємо тільки ті колонки, які реально є в df
drop_cols = [col for col in ID_COLS + LEAKAGE_COLS if col in df.columns]

feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

print("Features:", feature_cols)
print("Target distribution:")
print(y.value_counts())




cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# CatBoost краще працює, коли categorical values не NaN
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("missing")

# Numeric NaN CatBoost обробляє сам, але можна залишити як є
print("Categorical columns:", cat_cols)




X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train distribution:")
print(y_train.value_counts())

print("\nValid distribution:")
print(y_valid.value_counts())




X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train distribution:")
print(y_train.value_counts())

print("\nValid distribution:")
print(y_valid.value_counts())




def make_binary_class_weights(y_binary, positive_boost=1.0):
    """
    y_binary має містити 0/1.
    Повертає class_weights у форматі [weight_for_0, weight_for_1].
    positive_boost додатково підсилює клас 1.
    """
    y_binary = pd.Series(y_binary)
    counts = y_binary.value_counts().to_dict()

    n = len(y_binary)
    n_classes = 2

    w0 = n / (n_classes * counts.get(0, 1))
    w1 = n / (n_classes * counts.get(1, 1))

    w1 *= positive_boost

    return [w0, w1]




class HierarchicalSegmentModel:
    def __init__(
        self,
        cat_cols=None,
        stage1_positive_boost=1.3,
        stage2_hnwi_boost=2.0,
        random_state=42
    ):
        self.cat_cols = cat_cols or []
        self.stage1_positive_boost = stage1_positive_boost
        self.stage2_hnwi_boost = stage2_hnwi_boost
        self.random_state = random_state

        self.stage1_model = None
        self.stage2_model = None
        self.feature_cols = None

    def _prepare_X(self, X):
        X = X.copy()

        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna("missing")

        return X

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.feature_cols = X_train.columns.tolist()

        X_train = self._prepare_X(X_train)

        if X_valid is not None:
            X_valid = self._prepare_X(X_valid)

        # -----------------------------
        # Stage 1: MASS vs WEALTHY
        # -----------------------------
        y1_train = (y_train != "MASS").astype(int)

        stage1_weights = make_binary_class_weights(
            y1_train,
            positive_boost=self.stage1_positive_boost
        )

        self.stage1_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=2000,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=15,
            random_strength=2,
            bagging_temperature=1,
            class_weights=stage1_weights,
            random_seed=self.random_state,
            verbose=100,
            early_stopping_rounds=150
        )

        train_pool_1 = Pool(
            X_train,
            y1_train,
            cat_features=self.cat_cols
        )

        eval_set_1 = None
        if X_valid is not None and y_valid is not None:
            y1_valid = (y_valid != "MASS").astype(int)
            eval_set_1 = Pool(
                X_valid,
                y1_valid,
                cat_features=self.cat_cols
            )

        self.stage1_model.fit(
            train_pool_1,
            eval_set=eval_set_1,
            use_best_model=True if eval_set_1 is not None else False
        )

        # -----------------------------
        # Stage 2: PREM vs HNWI
        # -----------------------------
        wealthy_mask_train = y_train.isin(["PREM", "HNWI"])

        X2_train = X_train.loc[wealthy_mask_train].copy()
        y2_train_raw = y_train.loc[wealthy_mask_train].copy()

        # PREM = 0, HNWI = 1
        y2_train = (y2_train_raw == "HNWI").astype(int)

        stage2_weights = make_binary_class_weights(
            y2_train,
            positive_boost=self.stage2_hnwi_boost
        )

        self.stage2_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=2000,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=20,
            random_strength=3,
            bagging_temperature=1,
            class_weights=stage2_weights,
            random_seed=self.random_state,
            verbose=100,
            early_stopping_rounds=150
        )

        train_pool_2 = Pool(
            X2_train,
            y2_train,
            cat_features=self.cat_cols
        )

        eval_set_2 = None
        if X_valid is not None and y_valid is not None:
            wealthy_mask_valid = y_valid.isin(["PREM", "HNWI"])

            X2_valid = X_valid.loc[wealthy_mask_valid].copy()
            y2_valid_raw = y_valid.loc[wealthy_mask_valid].copy()
            y2_valid = (y2_valid_raw == "HNWI").astype(int)

            if y2_valid.nunique() == 2:
                eval_set_2 = Pool(
                    X2_valid,
                    y2_valid,
                    cat_features=self.cat_cols
                )

        self.stage2_model.fit(
            train_pool_2,
            eval_set=eval_set_2,
            use_best_model=True if eval_set_2 is not None else False
        )

        return self

    def predict_proba(self, X):
        X = X[self.feature_cols].copy()
        X = self._prepare_X(X)

        # Stage 1 probability: P(WEALTHY)
        p_wealthy = self.stage1_model.predict_proba(X)[:, 1]

        # Stage 2 probability: P(HNWI | WEALTHY)
        p_hnwi_given_wealthy = self.stage2_model.predict_proba(X)[:, 1]

        p_mass = 1 - p_wealthy
        p_hnwi = p_wealthy * p_hnwi_given_wealthy
        p_prem = p_wealthy * (1 - p_hnwi_given_wealthy)

        proba = pd.DataFrame({
            "P_MASS": p_mass,
            "P_PREM": p_prem,
            "P_HNWI": p_hnwi
        }, index=X.index)

        return proba

    def predict(
        self,
        X,
        hnwi_threshold=0.25,
        prem_threshold=0.40,
        mode="threshold"
    ):
        proba = self.predict_proba(X)

        if mode == "argmax":
            pred = proba[["P_MASS", "P_PREM", "P_HNWI"]].idxmax(axis=1)
            pred = pred.str.replace("P_", "", regex=False)
            return pred

        if mode == "threshold":
            pred = []

            for _, row in proba.iterrows():
                if row["P_HNWI"] >= hnwi_threshold:
                    pred.append("HNWI")
                elif row["P_PREM"] >= prem_threshold:
                    pred.append("PREM")
                else:
                    pred.append("MASS")

            return pd.Series(pred, index=X.index, name="predicted_segment")

        raise ValueError("mode must be either 'threshold' or 'argmax'")
    


model = HierarchicalSegmentModel(
    cat_cols=cat_cols,
    stage1_positive_boost=1.3,
    stage2_hnwi_boost=2.5,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    X_valid=X_valid,
    y_valid=y_valid
)




model = HierarchicalSegmentModel(
    cat_cols=cat_cols,
    stage1_positive_boost=1.3,
    stage2_hnwi_boost=2.5,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    X_valid=X_valid,
    y_valid=y_valid
)






def evaluate_thresholds(y_true, proba, hnwi_threshold, prem_threshold):
    preds = []

    for _, row in proba.iterrows():
        if row["P_HNWI"] >= hnwi_threshold:
            preds.append("HNWI")
        elif row["P_PREM"] >= prem_threshold:
            preds.append("PREM")
        else:
            preds.append("MASS")

    preds = pd.Series(preds, index=y_true.index)

    report = precision_recall_fscore_support(
        y_true,
        preds,
        labels=["MASS", "PREM", "HNWI"],
        zero_division=0
    )

    precision, recall, f1, support = report

    result = {
        "hnwi_threshold": hnwi_threshold,
        "prem_threshold": prem_threshold,

        "mass_precision": precision[0],
        "prem_precision": precision[1],
        "hnwi_precision": precision[2],

        "mass_recall": recall[0],
        "prem_recall": recall[1],
        "hnwi_recall": recall[2],

        "mass_f1": f1[0],
        "prem_f1": f1[1],
        "hnwi_f1": f1[2],

        "macro_f1": f1_score(y_true, preds, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),

        "n_pred_mass": (preds == "MASS").sum(),
        "n_pred_prem": (preds == "PREM").sum(),
        "n_pred_hnwi": (preds == "HNWI").sum()
    }

    # Бізнес-орієнтована функція якості.
    # HNWI важливіший, PREM другий, MASS третій.
    result["business_score"] = (
        0.45 * result["hnwi_recall"] +
        0.25 * result["prem_recall"] +
        0.15 * result["mass_recall"] +
        0.15 * result["macro_f1"]
    )

    return result


threshold_results = []

hnwi_grid = np.arange(0.05, 0.61, 0.025)
prem_grid = np.arange(0.20, 0.81, 0.025)

for hnwi_thr in hnwi_grid:
    for prem_thr in prem_grid:
        res = evaluate_thresholds(
            y_true=y_valid,
            proba=valid_proba,
            hnwi_threshold=hnwi_thr,
            prem_threshold=prem_thr
        )
        threshold_results.append(res)

threshold_results_df = pd.DataFrame(threshold_results)

threshold_results_df = threshold_results_df.sort_values(
    by="business_score",
    ascending=False
)

display(threshold_results_df.head(20))




best_row = threshold_results_df.iloc[0]

best_hnwi_threshold = best_row["hnwi_threshold"]
best_prem_threshold = best_row["prem_threshold"]

print("Best HNWI threshold:", best_hnwi_threshold)
print("Best PREM threshold:", best_prem_threshold)
print(best_row)







best_row = threshold_results_df.iloc[0]

best_hnwi_threshold = best_row["hnwi_threshold"]
best_prem_threshold = best_row["prem_threshold"]

print("Best HNWI threshold:", best_hnwi_threshold)
print("Best PREM threshold:", best_prem_threshold)
print(best_row)








def top_k_hnwi_analysis(y_true, proba, k_values=[25, 50, 100, 150, 200]):
    tmp = pd.DataFrame({
        "true_segment": y_true,
        "P_HNWI": proba["P_HNWI"],
        "P_PREM": proba["P_PREM"],
        "P_MASS": proba["P_MASS"]
    })

    tmp = tmp.sort_values("P_HNWI", ascending=False)

    total_hnwi = (tmp["true_segment"] == "HNWI").sum()

    rows = []

    for k in k_values:
        top_k = tmp.head(k)

        found_hnwi = (top_k["true_segment"] == "HNWI").sum()
        found_prem = (top_k["true_segment"] == "PREM").sum()
        found_mass = (top_k["true_segment"] == "MASS").sum()

        rows.append({
            "top_k": k,
            "found_hnwi": found_hnwi,
            "found_prem": found_prem,
            "found_mass": found_mass,
            "hnwi_recall_at_k": found_hnwi / total_hnwi if total_hnwi > 0 else np.nan,
            "hnwi_precision_at_k": found_hnwi / k
        })

    return pd.DataFrame(rows)


topk_df = top_k_hnwi_analysis(
    y_true=y_valid,
    proba=valid_proba,
    k_values=[10, 25, 50, 100, 150, 200]
)

display(topk_df)








valid_result = X_valid.copy()

valid_result["true_segment"] = y_valid
valid_result["P_MASS"] = valid_proba["P_MASS"]
valid_result["P_PREM"] = valid_proba["P_PREM"]
valid_result["P_HNWI"] = valid_proba["P_HNWI"]

valid_result["predicted_segment"] = y_pred_threshold

def make_business_group(row):
    if row["P_HNWI"] >= 0.50:
        return "Strong HNWI candidate"
    elif row["P_HNWI"] >= best_hnwi_threshold:
        return "Weak HNWI candidate"
    elif row["P_PREM"] >= best_prem_threshold:
        return "PREM candidate"
    else:
        return "Likely MASS"

valid_result["business_group"] = valid_result.apply(make_business_group, axis=1)

valid_result_sorted = valid_result.sort_values("P_HNWI", ascending=False)

display(valid_result_sorted.head(50))









final_model = HierarchicalSegmentModel(
    cat_cols=cat_cols,
    stage1_positive_boost=1.3,
    stage2_hnwi_boost=2.5,
    random_state=42
)

final_model.fit(
    X,
    y
)










X_external = df_external[feature_cols].copy()

external_proba = final_model.predict_proba(X_external)

external_pred = final_model.predict(
    X_external,
    hnwi_threshold=best_hnwi_threshold,
    prem_threshold=best_prem_threshold,
    mode="threshold"
)

external_result = df_external.copy()

external_result["P_MASS"] = external_proba["P_MASS"].values
external_result["P_PREM"] = external_proba["P_PREM"].values
external_result["P_HNWI"] = external_proba["P_HNWI"].values

external_result["predicted_segment"] = external_pred.values

external_result["business_group"] = external_result.apply(
    lambda row: (
        "Strong HNWI candidate" if row["P_HNWI"] >= 0.50 else
        "Weak HNWI candidate" if row["P_HNWI"] >= best_hnwi_threshold else
        "PREM candidate" if row["P_PREM"] >= best_prem_threshold else
        "Likely MASS"
    ),
    axis=1
)

external_result = external_result.sort_values("P_HNWI", ascending=False)

display(external_result.head(100))





import joblib

artifacts = {
    "model": final_model,
    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "best_hnwi_threshold": best_hnwi_threshold,
    "best_prem_threshold": best_prem_threshold
}

joblib.dump(artifacts, "hierarchical_hnwi_model.pkl")


artifacts = joblib.load("hierarchical_hnwi_model.pkl")

loaded_model = artifacts["model"]
feature_cols = artifacts["feature_cols"]
cat_cols = artifacts["cat_cols"]
best_hnwi_threshold = artifacts["best_hnwi_threshold"]
best_prem_threshold = artifacts["best_prem_threshold"]





