import numpy as np
import pandas as pd
import joblib


class TwoStageFXModel:
    def __init__(
        self,
        clf_model,
        reg_model,
        features_cols,
        cat_cols,
        num_medians=None,
        category_values=None,
        threshold=0.3,
        prediction_mode="expected"  # "expected" або "threshold"
    ):
        self.clf_model = clf_model
        self.reg_model = reg_model
        
        self.features_cols = list(features_cols)
        self.cat_cols = list(cat_cols)
        self.threshold = threshold
        self.prediction_mode = prediction_mode

        self.num_cols = [c for c in self.features_cols if c not in self.cat_cols]

        self.num_medians = num_medians if num_medians is not None else {}
        self.category_values = category_values if category_values is not None else {}

    def _prepare_X(self, X):
        X = X.copy()

        missing_cols = [c for c in self.features_cols if c not in X.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"У датафреймі немає потрібних фіч: {missing_cols}")

        X = X[self.features_cols].copy()

        # numerical columns
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

            fill_value = self.num_medians.get(c, 0)

            if pd.isna(fill_value):
                fill_value = 0

            X[c] = X[c].fillna(fill_value)

        # categorical columns
        for c in self.cat_cols:
            X[c] = X[c].astype("string").fillna("UNKNOWN")

            allowed_categories = self.category_values.get(c, None)

            if allowed_categories is not None:
                allowed_categories = list(allowed_categories)

                if "UNKNOWN" not in allowed_categories:
                    allowed_categories.append("UNKNOWN")

                # unseen categories -> UNKNOWN
                X[c] = X[c].where(X[c].isin(allowed_categories), "UNKNOWN")

                X[c] = pd.Categorical(
                    X[c],
                    categories=allowed_categories
                )
            else:
                X[c] = X[c].astype("category")

        return X

    def predict(self, X):
        """
        Returns:
            preds_final: фінальний FX-потенціал
            proba: P(FX > 0)
        """

        X_prep = self._prepare_X(X)

        proba = self.clf_model.predict_proba(X_prep)[:, 1]

        pred_log_cond = self.reg_model.predict(X_prep)
        pred_log_cond = np.clip(pred_log_cond, 0, None)

        pred_cond_amount = np.expm1(pred_log_cond)

        if self.prediction_mode == "expected":
            preds_final = proba * pred_cond_amount

        elif self.prediction_mode == "threshold":
            is_profitable = proba > self.threshold
            preds_final = np.zeros(len(X_prep))
            preds_final[is_profitable] = pred_cond_amount[is_profitable]

        else:
            raise ValueError("prediction_mode має бути 'expected' або 'threshold'")

        return preds_final, proba

    def predict_full(self, X):
        """
        Повертає dataframe з усіма виходами моделі.
        """

        X_prep = self._prepare_X(X)

        proba = self.clf_model.predict_proba(X_prep)[:, 1]

        pred_log_cond = self.reg_model.predict(X_prep)
        pred_log_cond = np.clip(pred_log_cond, 0, None)

        pred_cond_amount = np.expm1(pred_log_cond)

        expected_pred = proba * pred_cond_amount

        hard_pred = np.zeros(len(X_prep))
        is_profitable = proba > self.threshold
        hard_pred[is_profitable] = pred_cond_amount[is_profitable]

        result = X.copy()

        result["PROB_TO_FX"] = proba
        result["FX_COND_PRED"] = pred_cond_amount
        result["FX_EXPECTED"] = expected_pred
        result["FX_HARD_PRED"] = hard_pred

        if self.prediction_mode == "expected":
            result["FX_POTENTIAL"] = expected_pred
        else:
            result["FX_POTENTIAL"] = hard_pred

        return result
    

MODEL_PATH = r"C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Commissions_FX.pkl"

num_cols = [c for c in final_features if c not in cat_cols]

num_medians = {}

for c in num_cols:
    if c in df_train.columns:
        num_medians[c] = pd.to_numeric(df_train[c], errors="coerce").median()
    else:
        num_medians[c] = 0

category_values = {}

for c in cat_cols:
    if c in X_train_clf.columns:
        if str(X_train_clf[c].dtype) == "category":
            cats = list(X_train_clf[c].cat.categories.astype(str))
        else:
            cats = list(X_train_clf[c].astype("string").fillna("UNKNOWN").unique())

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
    threshold=0.3,
    prediction_mode="expected"   # краще для potential-моделі
)

joblib.dump(fx_model, MODEL_PATH)

print(f"FX model saved to: {MODEL_PATH}")










import joblib
import numpy as np
import pandas as pd

MODEL_PATH = r"C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Commissions_FX.pkl"

model = joblib.load(MODEL_PATH)

preds, proba = model.predict(df)

df["FX_POTENTIAL"] = preds
df["PROB_TO_FX"] = proba

df.head()