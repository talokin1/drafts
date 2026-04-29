import joblib

# Твій код навчання...
# reg.fit(...)

# 1. Зберігаємо саму модель
model_filename = 'lgbm_profit_model.joblib'
joblib.dump(reg, model_filename)

# 2. Зберігаємо список категоріальних колонок (це критично для інференсу!)
# Щоб при передбаченні ми точно знали, які колонки конвертувати в 'category'
cat_cols_filename = 'lgbm_cat_cols.joblib'
joblib.dump(cat_cols, cat_cols_filename)

print(f"Модель успішно збережена у {model_filename}")


import numpy as np
import pandas as pd
import joblib

class ProfitPredictor:
    def __init__(self, model_path: str, cat_cols_path: str):
        """
        Ініціалізує предиктор, завантажуючи модель та метадані.
        """
        self.model = joblib.load(model_path)
        self.cat_cols = joblib.load(cat_cols_path)
        
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Робить прогноз для нових даних і повертає результат у реальному масштабі.
        """
        # Створюємо копію, щоб не змінювати оригінальний датафрейм
        X_processed = X_new.copy()
        
        # 1. Перетворення категоріальних ознак
        # LightGBM вимагає, щоб категоріальні колонки мали тип 'category'
        for c in self.cat_cols:
            if c in X_processed.columns:
                X_processed[c] = X_processed[c].astype("category")
                
        # 2. Отримання прогнозу (результат буде в просторі log1p)
        y_pred_log = self.model.predict(X_processed)
        
        # 3. Зворотне перетворення (експоненціювання)
        y_pred_real = np.expm1(y_pred_log)
        
        return y_pred_real

# ==========================================
# ПРИКЛАД ВИКОРИСТАННЯ (ІНФЕРЕНС)
# ==========================================

if __name__ == "__main__":
    # Уявимо, що це твої нові дані, які прийшли з бази
    # new_data = pd.read_csv('new_customers.csv')
    
    # Для прикладу створимо фіктивний DataFrame
    # new_data = pd.DataFrame({'feature1': [10, 20], 'cat_feature': ['A', 'B']})
    
    # Ініціалізуємо наш клас
    predictor = ProfitPredictor(
        model_path='lgbm_profit_model.joblib',
        cat_cols_path='lgbm_cat_cols.joblib'
    )
    
    # Робимо прогноз
    # predictions = predictor.predict(new_data)
    # print("Прогнозований прибуток:", predictions)