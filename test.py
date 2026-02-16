import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, average_precision_score
import matplotlib.pyplot as plt

# ==========================================
# 1. Custom Two-Stage Model Class
# ==========================================
class TwoStageLGBM(BaseEstimator, RegressorMixin):
    def __init__(self, clf_params=None, reg_params=None, zero_threshold=0.5):
        """
        zero_threshold: поріг ймовірності класифікатора. 
        Якщо P(non-zero) < threshold, ставимо 0.
        """
        self.clf_params = clf_params if clf_params else {}
        self.reg_params = reg_params if reg_params else {}
        self.zero_threshold = zero_threshold
        
        self.classifier = None
        self.regressor = None
        self.is_fitted = False

    def fit(self, X, y, cat_features=None):
        # --- Stage 1: Classification (Zero vs Non-Zero) ---
        y_binary = (y > 0).astype(int)
        
        print(f"Training Classifier... (Balance: {y_binary.value_counts(normalize=True).to_dict()})")
        
        self.classifier = lgb.LGBMClassifier(**self.clf_params)
        self.classifier.fit(
            X, y_binary, 
            categorical_feature=cat_features if cat_features else 'auto',
            verbose=False
        )

        # --- Stage 2: Regression (Magnitude for Non-Zeros) ---
        # Вибираємо тільки ті рядки, де реальне значення > 0
        mask_nonzero = y > 0
        X_nonzero = X.loc[mask_nonzero]
        y_nonzero_log = np.log1p(y[mask_nonzero]) # Log-transform target
        
        print(f"Training Regressor on {len(X_nonzero)} non-zero samples...")
        
        self.regressor = lgb.LGBMRegressor(**self.reg_params)
        self.regressor.fit(
            X_nonzero, y_nonzero_log,
            categorical_feature=cat_features if cat_features else 'auto',
            verbose=False
        )
        
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Model not fitted yet!")
            
        # 1. Predict probability of being non-zero
        prob_nonzero = self.classifier.predict_proba(X)[:, 1]
        
        # 2. Predict value (in log scale) assuming it is non-zero
        pred_log = self.regressor.predict(X)
        pred_value = np.expm1(pred_log) # Inverse log transform
        
        # 3. Combine strategies
        # Варіант А: Жорсткий поріг (Hard Threshold)
        final_pred = np.where(prob_nonzero >= self.zero_threshold, pred_value, 0)
        
        # Варіант Б (можна розкоментувати): Очікуване значення (Expected Value)
        # final_pred = prob_nonzero * pred_value 
        
        return final_pred, prob_nonzero

# ==========================================
# 2. Setup & Training
# ==========================================

# Припустимо, що df - це твій датафрейм, а цільова колонка 'Assets'
# X = df.drop('Assets', axis=1)
# y = df['Assets']

# *ВАЖЛИВО*: Стратифікація при спліті має бути по бінарній цілі, 
# бо клас 1 дуже малий.
y_binary_strat = (y > 0).astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_binary_strat
)

# Визначаємо категоріальні колонки
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

# Параметри моделей
clf_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'class_weight': 'balanced', # КРИТИЧНО ВАЖЛИВО для 30к vs 1.8к
    'random_state': 42,
    'n_jobs': -1
}

reg_params = {
    'objective': 'regression', # Або 'huber' для стійкості до викидів
    'metric': 'mae',
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1
}

# Ініціалізація та навчання
model = TwoStageLGBM(clf_params, reg_params, zero_threshold=0.5)
model.fit(X_train, y_train, cat_features=cat_cols)

# ==========================================
# 3. Evaluation
# ==========================================

# Отримуємо прогнози
y_pred, prob_nonzero = model.predict(X_val)

# --- Metrics ---
print("="*60)
print("STAGE 1: Classification Metrics (Zero vs Non-Zero)")
print("="*60)
y_val_binary = (y_val > 0).astype(int)
print(f"ROC AUC: {roc_auc_score(y_val_binary, prob_nonzero):.4f}")
print(f"PR AUC : {average_precision_score(y_val_binary, prob_nonzero):.4f} (Focus on this!)")

print("\n" + "="*60)
print("STAGE 2: Final Regression Metrics (Combined)")
print("="*60)
print(f"MAE: {mean_absolute_error(y_val, y_pred):,.2f}")
print(f"R2 : {r2_score(y_val, y_pred):.4f}")

# --- Plotting ---
plt.figure(figsize=(12, 5))

# Plot 1: True vs Predicted (Log Scale for visibility)
plt.subplot(1, 2, 1)
plt.scatter(np.log1p(y_val), np.log1p(y_pred), alpha=0.3, s=10)
plt.plot([0, np.max(np.log1p(y_val))], [0, np.max(np.log1p(y_val))], 'r--')
plt.xlabel('True Values (log1p)')
plt.ylabel('Predicted Values (log1p)')
plt.title('True vs Predicted (Log Scale)')

# Plot 2: Residuals for Non-Zero predictions
mask_val_nonzero = y_val > 0
plt.subplot(1, 2, 2)
residuals = np.log1p(y_val[mask_val_nonzero]) - np.log1p(y_pred[mask_val_nonzero])
plt.hist(residuals, bins=50, alpha=0.7)
plt.title('Residuals Distribution (Only Non-Zeros)')
plt.xlabel('Log Error')

plt.tight_layout()
plt.show()