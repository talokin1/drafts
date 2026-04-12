class TwoStageFXModel:
    def __init__(
        self,
        clf_model,
        reg_model,
        cat_cols,
        features_cols,
        threshold=0.3
    ):
        self.clf_model = clf_model
        self.reg_model = reg_model
        self.cat_cols = cat_cols
        self.features_cols = features_cols
        self.threshold = threshold

    def _prepare_X(self, X):
        X = X.copy()
        X = X[self.features_cols]

        for c in self.cat_cols:
            X[c] = X[c].astype("category")

        return X

    def predict(self, X):
        X_prep = self._prepare_X(X)

        # 1. classification
        proba = self.clf_model.predict_proba(X_prep)[:, 1]
        is_profitable = proba > self.threshold

        # 2. regression
        preds = np.zeros(len(X_prep))

        if is_profitable.sum() > 0:
            preds[is_profitable] = self.reg_model.predict(X_prep[is_profitable])

        # 3. back to original scale
        preds_final = np.expm1(preds)

        return preds_final, proba
    

fx_model = TwoStageFXModel(
    clf_model=clf_binary,
    reg_model=reg,
    cat_cols=cat_cols,
    features_cols=final_features,
    threshold=0.3   # або той, що підбереш
)

import joblib

joblib.dump({
    "model": fx_model,
    "fitted_params": fitted_params   # щоб build_features повторити
}, "fx_two_stage_model.pkl")


import joblib

# load
bundle = joblib.load("fx_two_stage_model.pkl")

model = bundle["model"]
fitted_params = bundle["fitted_params"]

# preprocess
df_full_proc, _ = build_features(df_full, fitted_params)

# predict
preds, proba = model.predict(df_full_proc)

# фінальний результат
df_result = pd.DataFrame({
    "IDENTIFYCODE": df_full.index,
    "FX_POTENTIAL": preds,
    "FX_PROBA": proba
})


preds, proba = fx_model.predict(df_val_proc)

y_true = df_val[TARGET_NAME].values

validation_results = pd.DataFrame({
    "IDENTIFYCODE": df_val.index,
    "True": y_true,
    "Predicted": preds,
    "Proba": proba,
    "Is_Profitable_Pred": (proba > fx_model.threshold).astype(int),
    "Is_Profitable_True": (y_true > 0).astype(int)
})