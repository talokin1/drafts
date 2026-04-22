class LiabilitiesLargeModel:
    def __init__(self, classifier, regressor, cat_cols, feature_cols, target_threshold, classification_threshold):
        self.classifier = classifier
        self.regressor = regressor
        self.cat_cols = cat_cols
        self.feature_cols = feature_cols
        # Зберігаємо просто для довідки
        self.target_threshold = target_threshold 
        # А ЦЕЙ поріг будемо реально використовувати
        self.classification_threshold = classification_threshold 

    def predict(self, X):
        X_pred = X.copy()
        X_pred = X_pred[self.feature_cols]

        for c in self.cat_cols:
            X_pred[c] = X_pred[c].astype("category")

        # 1. ЗАМІНА: Беремо ймовірності замість жорстких класів
        class_proba = self.classifier.predict_proba(X_pred)[:, 1]
        
        # 2. ЗАМІНА: Застосовуємо наш кастомний поріг
        class_preds = (class_proba >= self.classification_threshold).astype(int)

        # Далі все залишається без змін
        reg_preds_log = self.regressor.predict(X_pred)
        reg_preds = np.exp1m(reg_preds_log)

        final_predictions = np.where(class_preds == 1, reg_preds, 0)
        return final_predictions

# Ініціалізація моделі
two_stage_model = LiabilitiesLargeModel(
    classifier=clf,
    regressor=reg,
    cat_cols=cat_cols,
    feature_cols=X_train.columns.to_list(),
    target_threshold=THRESHOLD,
    classification_threshold=CLASSIFICATION_TRESHOLD # Передаємо наш поріг 0.467!
)