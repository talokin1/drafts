class TransactionsModel:
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

        # E[y_log]
        medians = np.array([
            self.bucket_medians[i] for i in range(len(self.bucket_medians))
        ])
        y_log_pred = probs @ medians

        # повертаємо в original scale
        y_pred = np.expm1(y_log_pred)
        y_pred = np.clip(y_pred, 0, None)

        return y_pred
    

cat_values = {
    c: X_train[c].cat.categories
    for c in cat_cols
}

import joblib

wrapped_model = TransactionsModel(
    model=clf_model,
    bucket_medians=bucket_medians.to_dict(),
    cat_cols=cat_cols,
    cat_values=cat_values
)

joblib.dump(wrapped_model, "transactions_model.pkl")