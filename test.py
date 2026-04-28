import numpy as np
import pandas as pd
import joblib

class LiabilitiesSmallModel:
    """Модель для мас-маркету (MICRO, SMALL). Використовує MAE/L1 регресор."""
    def __init__(self, classifier, regressor, cat_cols, feature_cols, classification_threshold=0.5):
        self.classifier = classifier
        self.regressor = regressor
        self.cat_cols = cat_cols
        self.feature_cols = feature_cols
        self.classification_threshold = classification_threshold

    def predict(self, X):
        if X.empty:
            return np.array([])
            
        # Залізно відбираємо тільки ті фічі, на яких модель навчалася (наприклад, 173 шт)
        X_pred = X[self.feature_cols].copy()

        # Конвертація категоріальних змінних
        for c in self.cat_cols:
            if c in X_pred.columns:
                X_pred[c] = X_pred[c].astype("category")

        # Stage 1: Класифікатор з кастомним порогом
        class_proba = self.classifier.predict_proba(X_pred)[:, 1]
        class_preds = (class_proba >= self.classification_threshold).astype(int)

        # Stage 2: Регресор (повернення з log1p)
        reg_preds_log = self.regressor.predict(X_pred)
        reg_preds = np.expm1(reg_preds_log)

        # Об'єднання
        final_predictions = np.where(class_preds == 1, reg_preds, 0)
        return final_predictions


class LiabilitiesLargeModel:
    """Модель для великого бізнесу (LARGE). Використовує Tweedie регресор."""
    def __init__(self, classifier, regressor, cat_cols, feature_cols, classification_threshold=0.5):
        self.classifier = classifier
        self.regressor = regressor
        self.cat_cols = cat_cols
        self.feature_cols = feature_cols
        self.classification_threshold = classification_threshold

    def predict(self, X):
        if X.empty:
            return np.array([])
            
        # Залізно відбираємо тільки ті фічі, на яких модель навчалася
        X_pred = X[self.feature_cols].copy()

        # Конвертація категоріальних змінних
        for c in self.cat_cols:
             if c in X_pred.columns:
                X_pred[c] = X_pred[c].astype("category")

        # Stage 1: Класифікатор з кастомним порогом
        class_proba = self.classifier.predict_proba(X_pred)[:, 1]
        class_preds = (class_proba >= self.classification_threshold).astype(int)

        # Stage 2: Регресор (повернення з log1p)
        reg_preds_log = self.regressor.predict(X_pred)
        reg_preds = np.expm1(reg_preds_log)

        # Об'єднання
        final_predictions = np.where(class_preds == 1, reg_preds, 0)
        return final_predictions
    

# ЗБЕРЕЖЕННЯ МОДЕЛІ ДЛЯ БІДНИХ
small_model = LiabilitiesSmallModel(
    classifier=clf_small, # твій навчений класифікатор
    regressor=reg_small,  # твій навчений регресор (з MAE)
    cat_cols=cat_cols, 
    feature_cols=X_train.columns.to_list(), # ФІКСУЄМО 173 ФІЧІ
    classification_threshold=CLASSIFICATION_TRESHOLD # наприклад, 0.5 або 0.41
)
joblib.dump(small_model, r'...\Liabilities_SMALL.pkl')

# ЗБЕРЕЖЕННЯ МОДЕЛІ ДЛЯ БАГАТИХ
large_model = LiabilitiesLargeModel(
    classifier=clf_large, 
    regressor=reg_large,  # твій навчений регресор (з Tweedie)
    cat_cols=cat_cols,
    feature_cols=X_train.columns.to_list(), # ФІКСУЄМО 173 ФІЧІ
    classification_threshold=CLASSIFICATION_TRESHOLD 
)
joblib.dump(large_model, r'...\Liabilities_LARGE.pkl')