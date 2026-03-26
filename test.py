class AssetsBucketModel:
    def __init__(self, model, bucket_medians, cat_cols, cat_values=None):
        self.model = model
        self.bucket_medians = bucket_medians
        self.cat_cols = cat_cols
        self.cat_values = cat_values

    def predict(self, X):
        X = X.copy()

        # категоріальні фічі
        for c in self.cat_cols:
            if self.cat_values:
                X[c] = pd.Categorical(X[c], categories=self.cat_values[c])
            else:
                X[c] = X[c].astype("category")

        probs = self.model.predict_proba(X)

        # E[Y] через ймовірності
        y_pred = np.zeros(len(X))
        for i in range(len(self.bucket_medians)):
            y_pred += probs[:, i] * self.bucket_medians[i]

        return y_pred
    

import joblib

wrapped_model = AssetsBucketModel(
    model=clf_model,
    bucket_medians=bucket_medians,
    cat_cols=cat_cols,
    cat_values=cat_values
)

joblib.dump(wrapped_model, "assets_bucket_model.pkl")