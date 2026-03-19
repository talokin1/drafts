import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, log_loss

# ==========================================
# 1. Визначаємо межу (Threshold)
# ==========================================
# Дивлячись на твою гістограму (image_f0d2e4.png), "провал" між двома горбами 
# знаходиться приблизно на рівні логарифма 6.0 - 6.5. 
# Давай візьмемо 6.2 як точку розрізу (можеш підібрати оптимальну потім).
THRESHOLD_LOG = 6.2 

# Створюємо бінарні таргети для Роутера
y_train_class = (y_train_log > THRESHOLD_LOG).astype(int)
y_val_class = (y_val_log > THRESHOLD_LOG).astype(int)

# ==========================================
# 2. Навчаємо Router (Класифікатор)
# ==========================================
print("--- Training Router Classifier ---")
clf_router = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=30,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf_router.fit(
    X_train_final, 
    y_train_class,
    eval_set=[(X_val_final, y_val_class)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

# ==========================================
# 3. Розбиваємо дані для Регресорів
# ==========================================
# Тренувальна вибірка
mask_train_mass = y_train_log <= THRESHOLD_LOG
mask_train_vip = y_train_log > THRESHOLD_LOG

X_train_mass, y_train_mass = X_train_final[mask_train_mass], y_train_log[mask_train_mass]
X_train_vip, y_train_vip = X_train_final[mask_train_vip], y_train_log[mask_train_vip]

# Валідаційна вибірка (справжня, для early stopping регресорів)
mask_val_mass = y_val_log <= THRESHOLD_LOG
mask_val_vip = y_val_log > THRESHOLD_LOG

X_val_mass, y_val_mass_log = X_val_final[mask_val_mass], y_val_log[mask_val_mass]
X_val_vip, y_val_vip_log = X_val_final[mask_val_vip], y_val_log[mask_val_vip]

# ==========================================
# 4. Навчаємо Регресор 1 (Mass сегмент)
# ==========================================
print("\n--- Training Mass Regressor (y <= threshold) ---")
reg_mass = lgb.LGBMRegressor(
    objective="regression_l1", 
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=20, # Може бути менше, бо тут багато даних
    random_state=RANDOM_STATE
)
reg_mass.fit(
    X_train_mass, y_train_mass,
    eval_set=[(X_val_mass, y_val_mass_log)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)

# ==========================================
# 5. Навчаємо Регресор 2 (VIP сегмент)
# ==========================================
print("\n--- Training VIP Regressor (y > threshold) ---")
reg_vip = lgb.LGBMRegressor(
    objective="regression_l1", 
    n_estimators=2000,
    learning_rate=0.01, # Ще менший крок, бо вибірка менша і шумніша
    num_leaves=10,      # Менше листя, щоб не перенавчитись на малому горбі
    min_child_samples=10, 
    random_state=RANDOM_STATE
)
reg_vip.fit(
    X_train_vip, y_train_vip,
    eval_set=[(X_val_vip, y_val_vip_log)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)

# ==========================================
# 6. INFERENCE (Прогноз на валідації)
# ==========================================
print("\n--- Inference and Evaluation ---")
# 6.1. Роутер передбачає ймовірність, що клієнт VIP
p_vip_val = clf_router.predict_proba(X_val_final)[:, 1]

# 6.2. Обидва регресори роблять прогноз для ВСІХ клієнтів у валідації
pred_mass_log = reg_mass.predict(X_val_final)
pred_vip_log = reg_vip.predict(X_val_final)

# 6.3. Вибір стратегії об'єднання
# СТРАТЕГІЯ "HARD ROUTING": Якщо ймовірність > 0.5, беремо прогноз VIP, інакше Mass
# (Зазвичай краще працює для оптимізації MAE / медіани)
pred_final_log_hard = np.where(p_vip_val > 0.5, pred_vip_log, pred_mass_log)

# Переводимо в оригінальні гроші
pred_final_original = np.expm1(pred_final_log_hard)
y_val_original = np.expm1(y_val_log)

# Оцінка
final_mae = mean_absolute_error(y_val_original, pred_final_original)
print(f"FINAL ORIGINAL MAE (Two-Stage Model): {final_mae:,.2f}")