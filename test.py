class TwoStageIncomeModel:
    def __init__(
        self,
        clf_model,
        reg_model,
        bucket_medians,
        cat_cols,
        features_cols,
        threshold=0.3,
        cat_values=None
    ):
        self.clf_model = clf_model
        self.reg_model = reg_model
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

        # --- Stage 1: classification
        probs = self.clf_model.predict_proba(X)[:, 1]
        mask = probs > self.threshold

        # --- Stage 2: regression
        y_pred = np.zeros(len(X))

        if mask.sum() > 0:
            X_selected = X[mask]

            probs_reg = self.reg_model.predict_proba(X_selected)

            y_pred_selected = np.zeros(len(X_selected))
            for i in range(len(self.bucket_medians)):
                y_pred_selected += probs_reg[:, i] * self.bucket_medians[i]

            y_pred[mask] = y_pred_selected

        return y_pred

    def predict_proba_clf(self, X):
        X = self._prepare_X(X)
        return self.clf_model.predict_proba(X)[:, 1]
    

cat_values = {
    c: X_train_clf[c].cat.categories for c in cat_cols
}

wrapped_model = TwoStageIncomeModel(
    clf_model=clf_model,
    reg_model=reg_model,
    bucket_medians=bucket_medians,
    cat_cols=cat_cols,
    features_cols=final_features,
    threshold=0.3,
    cat_values=cat_values
)

joblib.dump(
    wrapped_model,
    r"...\TwoStage_Accounts_Model.pkl"
)











y_pred_final = wrapped_model.predict(X_val_clf)

validation_results = pd.DataFrame({
    "IDENTIFYCODE": X_val_clf.index,
    "True_Value": y_val,
    "Predicted": y_pred_final,
    "CLF_Prob": wrapped_model.predict_proba_clf(X_val_clf)
})


print("MAE FINAL:", mean_absolute_error(y_val, y_pred_final))

# тільки для тих, кого clf пропустив
mask = y_pred_final > 0
print("MAE NON-ZERO:", mean_absolute_error(y_val[mask], y_pred_final[mask]))