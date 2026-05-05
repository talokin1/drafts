# ... (Твій код навчання clf та reg залишається без змін) ...

print("Розрахунок фінальних прогнозів (Expected Value)...")

# 1. Отримуємо ймовірності від класифікатора
prob_val = clf.predict_proba(X_val)[:, 1]

# 2. Отримуємо прогнози від регресора (з поверненням з логарифма)
val_reg_preds_log = reg.predict(X_val)
val_reg_preds = np.expm1(val_reg_preds_log)
val_reg_preds = np.clip(val_reg_preds, 0, None)

# 3. Капінг екстремальних значень (як у твоєму коді)
CAP_Q = 0.97
global_cap = np.quantile(y_train_raw[y_train_raw >= THRESHOLD], CAP_Q)
val_reg_preds_capped = np.clip(val_reg_preds, 0, global_cap)

# ==========================================
# НОВА ЛОГІКА: Математичне сподівання
# ==========================================
# E[Y] = P(Y > 0) * E[Y | Y > 0]
y_pred_expected = prob_val * val_reg_preds_capped

# Відсікаємо зовсім малі ймовірності (шум), щоб не давати копійчані прогнози мертвим клієнтам
MIN_PROBA_TO_PREDICT = 0.05 
y_pred_final = np.where(prob_val >= MIN_PROBA_TO_PREDICT, y_pred_expected, 0)

# ==========================================
# Оцінка
# ==========================================
y_val_final_log = np.log1p(y_val_raw)
y_pred_final_log = np.log1p(y_pred_final)

mae = mean_absolute_error(y_val_raw, y_pred_final)
medae = median_absolute_error(y_val_raw, y_pred_final)
r2 = r2_score(y_val_raw, y_pred_final)

mae_log = mean_absolute_error(y_val_final_log, y_pred_final_log)
r2_log = r2_score(y_val_final_log, y_pred_final_log)

print("=" * 70)
print("FINAL COMBINED PIPELINE METRICS (EXPECTED VALUE)")
print("=" * 70)
print(f"MAE       : {mae:,.2f}")
print(f"MedAE     : {medae:,.2f}")
print(f"R2        : {r2:.4f}")
print(f"MAE_log   : {mae_log:.5f}")
print(f"R2_log    : {r2_log:.5f}")
print("-" * 70)














import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

# Фіксуємо змінні
RANDOM_STATE = 42

# 1. Очищення цільової змінної (без логарифмування!)
y_clean = np.clip(y, a_min=0, a_max=None)

# 2. Розбиття даних. 
# Стратифікуємо по факту наявності доходу, щоб нулі розподілилися рівномірно.
y_binary = (y_clean > 0).astype(int) 

X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    X_, y_clean, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
)

# 3. Обробка категоріальних ознак
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

print("Навчання Tweedie Regressor...")

# 4. Ініціалізація моделі
reg_tweedie = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,  # Ключовий параметр! Від 1.0 (Пуассон) до 2.0 (Гамма). 1.5 - хороший старт.
    metric="rmse",               # Для Tweedie RMSE зазвичай дає кращі градієнти
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    categorical_feature=cat_cols,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# 5. Навчання (передаємо СИРІ дані з нулями, без np.log1p)
reg_tweedie.fit(
    X_train, y_train_raw,
    eval_set=[(X_val, y_val_raw)],
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)

# 6. Прогнозування (модель сама повертає абсолютні невід'ємні значення)
val_preds_tweedie = reg_tweedie.predict(X_val)

# 7. Капінг (щоб один аномальний прогноз не зіпсував MAE)
CAP_Q = 0.97
global_cap = np.quantile(y_train_raw[y_train_raw > 0], CAP_Q)
y_pred_final = np.clip(val_preds_tweedie, 0, global_cap)

# ==========================================
# Оцінка
# ==========================================
y_val_final_log = np.log1p(y_val_raw)
y_pred_final_log = np.log1p(y_pred_final)

mae = mean_absolute_error(y_val_raw, y_pred_final)
medae = median_absolute_error(y_val_raw, y_pred_final)
r2 = r2_score(y_val_raw, y_pred_final)

mae_log = mean_absolute_error(y_val_final_log, y_pred_final_log)
r2_log = r2_score(y_val_final_log, y_pred_final_log)

print("=" * 70)
print("FINAL METRICS (TWEEDIE REGRESSION)")
print("=" * 70)
print(f"MAE       : {mae:,.2f}")
print(f"MedAE     : {medae:,.2f}")
print(f"R2        : {r2:.4f}")
print(f"MAE_log   : {mae_log:.5f}")
print(f"R2_log    : {r2_log:.5f}")
print("-" * 70)




