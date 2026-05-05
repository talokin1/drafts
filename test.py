import joblib
import numpy as np
import pandas as pd


class AssetsPotentialModel:
    def __init__(self, model_package):
        self.pkg = model_package

        self.feature_cols = model_package["feature_cols"]
        self.cat_cols = model_package["cat_cols"]
        self.cat_values = model_package["cat_values"]
        self.removed_corr_features = model_package.get("removed_corr_features", [])

        self.active_classifier = model_package["active_classifier"]
        self.bucket_classifier = model_package["bucket_classifier"]
        self.bucket_values = model_package["bucket_values"]

        self.tail_regressor = model_package.get("tail_regressor", None)
        self.tail_cap = model_package.get("tail_cap", None)

        self.threshold = model_package["best_active_probability_threshold"]

    def prepare_X(self, df):
        X_inf = df.copy()

        X_inf = X_inf.drop(columns=self.removed_corr_features, errors="ignore")

        for col in self.feature_cols:
            if col not in X_inf.columns:
                X_inf[col] = np.nan

        X_inf = X_inf[self.feature_cols].copy()

        for c in self.cat_cols:
            X_inf[c] = pd.Categorical(
                X_inf[c],
                categories=self.cat_values[c]
            )

        return X_inf

    def predict(self, df, return_details=False):
        X_inf = self.prepare_X(df)

        active_prob = self.active_classifier.predict_proba(X_inf)[:, 1]

        bucket_proba = self.bucket_classifier.predict_proba(X_inf)
        bucket_idx = np.argmax(bucket_proba, axis=1)

        amount_pred = bucket_proba @ self.bucket_values

        if self.tail_regressor is not None:
            top_bucket = len(self.bucket_values) - 1
            top_mask = bucket_idx == top_bucket

            if top_mask.sum() > 0:
                tail_pred = np.expm1(
                    self.tail_regressor.predict(X_inf.loc[top_mask])
                )
                tail_pred = np.clip(tail_pred, 0, self.tail_cap)

                amount_pred[top_mask] = (
                    0.5 * amount_pred[top_mask]
                    + 0.5 * tail_pred
                )

        final_pred = np.where(
            active_prob >= self.threshold,
            amount_pred,
            0
        )

        final_pred = np.clip(final_pred, 0, None)

        if not return_details:
            return final_pred

        return pd.DataFrame({
            "ASSETS_POTENTIAL": final_pred,
            "ASSETS_ACTIVE_PROB": active_prob,
            "ASSETS_BUCKET": bucket_idx,
            "ASSETS_IS_ACTIVE_PRED": (
                active_prob >= self.threshold
            ).astype(int)
        }, index=df.index)
    

model_package = joblib.load(r"C:\Projects\DS-450\assets_potential_bucket_model.joblib")

assets_model = AssetsPotentialModel(model_package)

pred_details = assets_model.predict(df, return_details=True)

df["ASSETS_POTENTIAL"] = pred_details["ASSETS_POTENTIAL"]
df["ASSETS_ACTIVE_PROB"] = pred_details["ASSETS_ACTIVE_PROB"]
df["ASSETS_BUCKET"] = pred_details["ASSETS_BUCKET"]
df["ASSETS_IS_ACTIVE_PRED"] = pred_details["ASSETS_IS_ACTIVE_PRED"]

df[[
    "ASSETS_POTENTIAL",
    "ASSETS_ACTIVE_PROB",
    "ASSETS_BUCKET",
    "ASSETS_IS_ACTIVE_PRED"
]].head()

df["ASSETS_POTENTIAL"] = assets_model.predict(df, return_details=False)