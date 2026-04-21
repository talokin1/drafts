import joblib
import numpy as np
import pandas as pd

class TwoStageAssetsIncomeModel:
    def __init__(self, classifier, regressor, cat_cols, feature_cols, classification_threshold=0.28):
        # Зберігаємо ОБИДВІ моделі
        self.clf = classifier
        self.reg = regressor
        self.cat_cols = cat_cols
        self.feature_cols = feature_cols
        self.threshold = classification_threshold

    def predict(self, X):
        X_infer = X.copy()
        X_infer = X_infer[self.feature_cols]
        
        # 1. Відновлюємо типи категоріальних ознак для LightGBM
        for c in self.cat_cols:
            X_infer[c] = X_infer[c].astype("category")
            
        # 2. Stage 1: Оцінка ймовірності активності
        probs = self.clf.predict_proba(X_infer)[:, 1]
        is_active = (probs >= self.threshold).astype(int)
        
        # 3. Stage 2: Прогноз об'єму (повертає логарифмовані значення)
        # Інженерна оптимізація: ми робимо predict для всіх, щоб не ламати індекси масивів, 
        # але для надвеликих батчів можна оптимізувати, викликаючи reg тільки для is_active == 1
        reg_preds_log = self.reg.predict(X_infer)
        
        # 4. Зворотне математичне перетворення (з log1p у звичайний простір)
        reg_preds_money = np.expm1(reg_preds_log)
        
        # 5. Об'єднання логіки (накладання маски класифікатора)
        final_preds = np.where(is_active == 1, reg_preds_money, 0)
        
        return final_preds

# ==========================================
# Ініціалізація та збереження
# ==========================================

# Створюємо екземпляр класу з нашими навченими моделями
final_pipeline = TwoStageAssetsIncomeModel(
    classifier=clf,
    regressor=reg,
    cat_cols=cat_cols,
    feature_cols=X_.columns.to_list(),
    classification_threshold=0.28
)

# Зберігаємо весь об'єкт
joblib.dump(final_pipeline, r"C:\Projects\(DS-450) Corp_potential_assets_model.pkl")