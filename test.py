class FXModel:
    def __init__(self, model, bucket_medians, cat_cols, cat_values=None):
        self.model = model
        self.bucket_medians = bucket_medians  # тут ЗБЕРІГАЄМО log-медіани
        self.cat_cols = cat_cols
        self.cat_values = cat_values

        # одразу підготуємо medians в original scale
        self.medians_orig = np.array([
            np.expm1(self.bucket_medians[i]) for i in range(len(self.bucket_medians))
        ])

    def predict(self, X):
        X = X.copy()

        # категоріальні
        for c in self.cat_cols:
            if self.cat_values:
                X[c] = pd.Categorical(X[c], categories=self.cat_values[c])
            else:
                X[c] = X[c].astype("category")

        probs = self.model.predict_proba(X)

        # E[y] вже в original scale
        y_pred = probs @ self.medians_orig
        y_pred = np.clip(y_pred, 0, None)

        return y_pred
    
cat_values = {
    c: X_train[c].cat.categories
    for c in cat_cols
}

import joblib

wrapped_model = FXModel(
    model=clf_model,
    bucket_medians=bucket_medians.to_dict(),  # лог-медіани
    cat_cols=cat_cols,
    cat_values=cat_values
)

joblib.dump(wrapped_model, "fx_model.pkl")