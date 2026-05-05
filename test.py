import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# ==========================================
# 1. ПІДГОТОВКА ДАНИХ
# ==========================================
RANDOM_STATE = 42

# Бізнес-логіка: все що вище 0 (або іншого малого значення) вважаємо активністю
ACTIVITY_THRESHOLD = 0.0  

y_clean = np.clip(y, a_min=0, a_max=None)
y_binary = (y_clean > ACTIVITY_THRESHOLD).astype(int)

X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    X_, y_clean, 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
    stratify=y_binary
)

# Створюємо y для класифікатора (0 або 1)
y_train_clf = (y_train_raw > ACTIVITY_THRESHOLD).astype(int)
y_val_clf = (y_val_raw > ACTIVITY_THRESHOLD).astype(int)

# Категоріальні фічі
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

# ==========================================
# 2. STAGE 1: КЛАСИФІКАТОР (Активний / Неактивний)
# ==========================================
print("Навчання Stage 1: Classifier...")

# Розрахунок балансу класів для scale_pos_weight
pos = y_train_clf.sum()
neg = len(y_train_clf) - pos
scale_weight = np.sqrt(neg/pos) if pos > 0 else 1.0

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    categorical_feature=cat_cols,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train, y_train_clf,
    eval_set=[(X_val, y_val_clf)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

# Прогнозуємо ймовірності для валідації
prob_val = clf.predict_proba(X_val)[:, 1]

# --- Підбір оптимального порогу по F1-score ---
# Це математично краще, ніж кастомний score, бо оптимізує гармонійне середнє 
# між Precision (щоб не було False Positives) та Recall (щоб знайти всіх активних)
best_f1 = 0
best_threshold = 0.5

for th in np.arange(0.1, 0.9, 0.02):
    y_pred_th = (prob_val >= th).astype(int)
    current_f1 = f1_score(y_val_clf, y_pred_th, zero_division=0)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = th

print(f"\nОптимальний поріг класифікації: {best_threshold:.2f} (F1 = {best_f1:.3f})")
print(classification_report(y_val_clf, (prob_val >= best_threshold).astype(int)))

# Фіксуємо бінарний прогноз для бізнесу
val_is_active_pred = (prob_val >= best_threshold).astype(int)

# ==========================================
# 3. STAGE 2: РЕГРЕСОР (Оцінка прибутку)
# ==========================================
print("\nНавчання Stage 2: Regressor...")

# ВАЖЛИВО: Регресор вчиться ТІЛЬКИ на активних клієнтах
mask_train_active = y_train_clf == 1
X_train_reg = X_train[mask_train_active]
# Логарифмуємо таргет, бо розподіл грошей завжди скошений вправо
y_train_reg_log = np.log1p(y_train_raw[mask_train_active]) 

mask_val_active = y_val_clf == 1
X_val_reg = X_val[mask_val_active]
y_val_reg_log = np.log1p(y_val_raw[mask_val_active])

reg = lgb.LGBMRegressor(
    objective="regression_l1", # L1 (MAE) стійкіший до викидів, ніж L2 (MSE)
    metric='mae',
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

reg.fit(
    X_train_reg, y_train_reg_log,
    eval_set=[(X_val_reg, y_val_reg_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)

# Прогнозуємо для ВСІХ клієнтів (повертаємося з логарифма)
val_reg_preds_log = reg.predict(X_val)
val_reg_preds = np.expm1(val_reg_preds_log)
val_reg_preds = np.clip(val_reg_preds, 0, None)

# Капінг для стабілізації (щоб 1 клієнт-мільярдер не ламав метрики)
CAP_Q = 0.97
global_cap = np.quantile(y_train_raw[y_train_raw > ACTIVITY_THRESHOLD], CAP_Q)
val_reg_preds_capped = np.clip(val_reg_preds, 0, global_cap)

# ==========================================
# 4. ОБ'ЄДНАННЯ МОДЕЛЕЙ (Математичне сподівання)
# ==========================================
# Якщо клієнт класифікований як неактивний (0) -> дохід 0
# Якщо активний (1) -> зважуємо прогноз регресора на ймовірність!
# Це захищає нас від того, що регресор "переоцінить" клієнта, в якому класифікатор сумнівався.
y_pred_final = val_is_active_pred * (prob_val * val_reg_preds_capped)


# ==========================================
# 5. ОЦІНКА ЯКОСТІ (МЕТРИКИ)
# ==========================================
y_val_final_log = np.log1p(y_val_raw)
y_pred_final_log = np.log1p(y_pred_final)

mae = mean_absolute_error(y_val_raw, y_pred_final)
medae = median_absolute_error(y_val_raw, y_pred_final)
r2 = r2_score(y_val_raw, y_pred_final)

mae_log = mean_absolute_error(y_val_final_log, y_pred_final_log)
medae_log = median_absolute_error(y_val_final_log, y_pred_final_log)
r2_log = r2_score(y_val_final_log, y_pred_final_log)

print("=" * 70)
print("FINAL COMBINED PIPELINE METRICS (CLASSIFIER + REGRESSOR)")
print("=" * 70)
print(f"MAE       : {mae:,.2f}")
print(f"MedAE     : {medae:,.2f}")
print(f"R2        : {r2:.4f}")
print(f"MAE_log   : {mae_log:.5f}")
print(f"MedAE_log : {medae_log:.5f}")
print(f"R2_log    : {r2_log:.5f}")
print("-" * 70)

# Для бізнесу ти можеш зібрати результат в таку таблицю:
# results_df = pd.DataFrame({
#     'is_active_prediction': val_is_active_pred, 
#     'predicted_income': y_pred_final
# })