import numpy as np
import pandas as pd
import joblib
import os


class TwoStageFXModel:
    def __init__(
        self,
        clf_model,
        reg_model,
        features_cols,
        cat_cols,
        num_medians,
        category_values,
        threshold=0.30,
        prediction_mode="expected"
    ):
        self.clf_model = clf_model
        self.reg_model = reg_model

        self.features_cols = list(features_cols)
        self.cat_cols = list(cat_cols)
        self.num_cols = [c for c in self.features_cols if c not in self.cat_cols]

        self.num_medians = num_medians
        self.category_values = category_values

        self.threshold = threshold
        self.prediction_mode = prediction_mode

    def _prepare_X(self, df):
        X = df.copy()

        missing_cols = [c for c in self.features_cols if c not in X.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"У df немає потрібних фіч: {missing_cols}")

        X = X[self.features_cols].copy()

        # numeric columns
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

            fill_value = self.num_medians.get(c, 0)
            if pd.isna(fill_value):
                fill_value = 0

            X[c] = X[c].fillna(fill_value)

        # categorical columns
        for c in self.cat_cols:
            X[c] = X[c].astype("string").fillna("UNKNOWN")

            cats = self.category_values.get(c, ["UNKNOWN"])
            cats = list(cats)

            if "UNKNOWN" not in cats:
                cats.append("UNKNOWN")

            # unseen categories -> UNKNOWN
            X[c] = X[c].where(X[c].isin(cats), "UNKNOWN")

            X[c] = pd.Categorical(X[c], categories=cats)

        return X

    def predict(self, df):
        X = self._prepare_X(df)

        # 1. ймовірність FX активності
        proba = self.clf_model.predict_proba(X)[:, 1]

        # 2. умовний прогноз суми FX, якщо клієнт активний
        pred_log = self.reg_model.predict(X)
        pred_log = np.clip(pred_log, 0, None)

        fx_cond_pred = np.expm1(pred_log)

        # 3. фінальний potential
        if self.prediction_mode == "expected":
            fx_potential = proba * fx_cond_pred

        elif self.prediction_mode == "threshold":
            fx_potential = np.zeros(len(df))
            mask = proba >= self.threshold
            fx_potential[mask] = fx_cond_pred[mask]

        else:
            raise ValueError("prediction_mode має бути 'expected' або 'threshold'")

        return fx_potential, proba

    def predict_full(self, df):
        X = self._prepare_X(df)

        proba = self.clf_model.predict_proba(X)[:, 1]

        pred_log = self.reg_model.predict(X)
        pred_log = np.clip(pred_log, 0, None)

        fx_cond_pred = np.expm1(pred_log)

        fx_expected = proba * fx_cond_pred

        fx_threshold = np.zeros(len(df))
        mask = proba >= self.threshold
        fx_threshold[mask] = fx_cond_pred[mask]

        result = df.copy()

        result["PROB_TO_FX"] = proba
        result["FX_COND_PRED"] = fx_cond_pred
        result["FX_EXPECTED"] = fx_expected
        result["FX_THRESHOLD_PRED"] = fx_threshold

        if self.prediction_mode == "expected":
            result["FX_POTENTIAL"] = fx_expected
        else:
            result["FX_POTENTIAL"] = fx_threshold

        return result
    

MODEL_PATH = r"C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Commissions_FX.pkl"

# numeric columns
num_cols = [c for c in final_features if c not in cat_cols]

# train medians for numeric columns
num_medians = {}

for c in num_cols:
    if c in df_train.columns:
        num_medians[c] = pd.to_numeric(df_train[c], errors="coerce").median()
    else:
        num_medians[c] = 0

# categories for categorical columns
category_values = {}

for c in cat_cols:
    if c in X_train_clf.columns:
        if str(X_train_clf[c].dtype) == "category":
            cats = list(X_train_clf[c].cat.categories.astype(str))
        else:
            cats = list(
                X_train_clf[c]
                .astype("string")
                .fillna("UNKNOWN")
                .unique()
            )

        if "UNKNOWN" not in cats:
            cats.append("UNKNOWN")

        category_values[c] = cats

fx_model = TwoStageFXModel(
    clf_model=clf_binary,
    reg_model=reg,
    features_cols=final_features,
    cat_cols=cat_cols,
    num_medians=num_medians,
    category_values=category_values,
    threshold=0.30,
    prediction_mode="expected"
)

joblib.dump(fx_model, MODEL_PATH)

print("Saved FX model to:")
print(MODEL_PATH)















import joblib
import numpy as np
import pandas as pd

MODEL_PATH = r"C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Commissions_FX.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded")

preds, proba = model.predict(df)

df["FX_POTENTIAL"] = preds
df["PROB_TO_FX"] = proba

df[["FX_POTENTIAL", "PROB_TO_FX"]].describe()