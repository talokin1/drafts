import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Припустимо, що у тебе вже є X_train, X_val, y_train_log, y_val_log
# Константи LightGBM
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1
}

# ==========================================
# КРОК 1: Навчаємо модель для середнього (Mu)
# ==========================================
print("Training Mean Model (Mu)...")
model_mu = lgb.LGBMRegressor(**LGB_PARAMS)
model_mu.fit(
    X_train, y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

# Отримуємо прогнози середнього
mu_train = model_mu.predict(X_train)
mu_val = model_mu.predict(X_val)

# ==========================================
# КРОК 2: Готуємо таргет для моделі Sigma
# ==========================================
# Ми хочемо передбачити величину помилки.
# Використовуємо логарифм квадрату залишків або модуль залишків.
# Найстабільніший варіант для interpretable sigma: модуль помилки.
# sigma ~ |y_true - y_pred|
resid_train = np.abs(y_train_log - mu_train)
resid_val = np.abs(y_val_log - mu_val)

print(f"Mean Residual on Train: {np.mean(resid_train):.4f}")

# ==========================================
# КРОК 3: Навчаємо модель для дисперсії (Sigma)
# ==========================================
print("Training Uncertainty Model (Sigma)...")
# Тут важливо змінити метрику, бо ми передбачаємо помилку
sigma_params = LGB_PARAMS.copy()
sigma_params['metric'] = 'mse' # MSE краще карає за великі промахи у передбаченні дисперсії

model_sigma = lgb.LGBMRegressor(**sigma_params)
model_sigma.fit(
    X_train, resid_train,
    eval_set=[(X_val, resid_val)],
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

sigma_val = model_sigma.predict(X_val)

# Гарантуємо, що sigma не від'ємна (хоча ReLU це робить, але для безпеки)
sigma_val = np.maximum(sigma_val, 1e-6)

# ==========================================
# КРОК 4: Формуємо Probabilistic Prediction
# ==========================================
# Prediction = Mu + (k * Sigma)
# k - це коефіцієнт "апетиту до ризику".
# k=1.0 ~ 84-й перцентиль (для норм. розп.)
# k=1.64 ~ 95-й перцентиль
# k=2.0 ~ 97.7-й перцентиль

K_FACTOR = 1.5  # Налаштуй цей параметр під бізнес-потребу!

y_pred_potential_log = mu_val + (sigma_val * K_FACTOR)

# Конвертуємо назад з логарифму
final_pred_mean = np.expm1(mu_val)
final_pred_potential = np.expm1(y_pred_potential_log)
y_true = np.expm1(y_val_log)

# ==========================================
# КРОК 5: Візуалізація результату
# ==========================================
plt.figure(figsize=(12, 6))

# Scatter plot: Reality vs Mean Prediction vs Potential Prediction
plt.scatter(y_val_log, mu_val, alpha=0.3, s=10, color='gray', label='Predicted Mean (Conservative)')
plt.scatter(y_val_log, y_pred_potential_log, alpha=0.3, s=10, color='red', label=f'Predicted Potential (Mu + {K_FACTOR}*Sigma)')

# Ідеальна лінія
mn, mx = y_val_log.min(), y_val_log.max()
plt.plot([mn, mx], [mn, mx], 'k--', lw=2, label='Identity')

plt.xlabel('True Value (log)')
plt.ylabel('Predicted Value (log)')
plt.title('Probabilistic Regression: Mean vs Potential')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Перевірка на прикладах
df_res = pd.DataFrame({
    'True': y_true,
    'Pred_Mean': final_pred_mean,
    'Pred_Sigma (log scale)': sigma_val,
    'Pred_Potential': final_pred_potential
}).round(2)

print("\nTOP 10 'Whales' by Potential (High Sigma):")
print(df_res.sort_values('Pred_Potential', ascending=False).head(10))

print("\nTOP 10 Conservative Clients (Low Sigma):")
print(df_res.sort_values('Pred_Sigma (log scale)', ascending=True).head(10))