import numpy as np
import pandas as pd


class TwoStageIncomeModel:
    def __init__(
        self,
        clf_binary,
        clf_multiclass,
        bucket_medians,
        cat_cols,
        features_cols,
        threshold=0.0,
        cat_values=None
    ):
        self.clf_binary = clf_binary
        self.clf_multiclass = clf_multiclass
        self.bucket_medians = bucket_medians
        self.cat_cols = cat_cols
        self.features_cols = features_cols
        self.threshold = threshold
        self.cat_values = cat_values

    def _prepare_X(self, X):
        X = X.copy()
        X = X[self.features_cols]

        for c in self.cat_cols:
            if self.cat_values:
                X[c] = pd.Categorical(X[c], categories=self.cat_values[c])
            else:
                X[c] = X[c].astype("category")

        return X

    def predict(self, X):
        X = self._prepare_X(X)

        # ---- 1. classifier
        p_income = self.clf_binary.predict_proba(X)[:, 1]

        # ---- 2. multiclass regression
        probs_multi = self.clf_multiclass.predict_proba(X)

        medians = np.array([
            self.bucket_medians[i]
            for i in range(len(self.bucket_medians))
        ])

        y_log_expected = probs_multi @ medians
        y_expected = np.expm1(y_log_expected)
        y_expected = np.clip(y_expected, 0, None)

        # ---- 3. final
        final_pred = p_income * y_expected

        # ---- 4. threshold (опціонально)
        if self.threshold > 0:
            final_pred[p_income < self.threshold] = 0

        return final_pred

    def predict_components(self, X):
        """
        Для дебагу / аналізу
        """
        X = self._prepare_X(X)

        p_income = self.clf_binary.predict_proba(X)[:, 1]
        probs_multi = self.clf_multiclass.predict_proba(X)

        medians = np.array([
            self.bucket_medians[i]
            for i in range(len(self.bucket_medians))
        ])

        y_expected = np.expm1(probs_multi @ medians)

        return p_income, y_expected
    

import joblib

model = joblib.load("two_stage_income_model.pkl")

preds = model.predict(df_new)

df_new["ACCOUNT_POTENTIAL"] = preds

cat_values = {
    c: X_train[c].cat.categories
    for c in cat_cols
}