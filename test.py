import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score

print("=" * 60)
print(" ДІАГНОСТИКА ПЕРЕНАВЧАННЯ (TRAIN vs VAL)")
print("=" * 60)

# ==========================================
# ЕТАП 1: Оцінка Класифікатора (Stage 1)
# ==========================================
# Прогнози для трейну та валідації
train_clf_probs = clf.predict_proba(X_train)[:, 1]
val_clf_probs = clf.predict_proba(X_val)[:, 1]

# Метрики
train_auc = roc_auc_score(y_train_clf, train_clf_probs)
val_auc = roc_auc_score(y_val_clf, val_clf_probs)

print(f"\n[Stage 1: Classifier]")
print(f"ROC-AUC  | Train: {train_auc:.4f} | Val: {val_auc:.4f} | Різниця: {train_auc - val_auc:.4f}")


# ==========================================
# ЕТАП 2: Оцінка Регресора (Stage 2)
# ==========================================
# Регресор навчався ТІЛЬКИ на активних клієнтах (де y >= THRESHOLD)
train_reg_preds_log = reg.predict(X_train_reg)
val_reg_preds_log = reg.predict(X_val_reg)

train_mae_reg = mean_absolute_error(y_train_reg_log, train_reg_preds_log)
val_mae_reg = mean_absolute_error(y_val_reg_log, val_reg_preds_log)

train_r2_reg = r2_score(y_train_reg_log, train_reg_preds_log)
val_r2_reg = r2_score(y_val_reg_log, val_reg_preds_log)

print(f"\n[Stage 2: Regressor (Only Active Clients, Log-Space)]")
print(f"MAE      | Train: {train_mae_reg:.4f} | Val: {val_mae_reg:.4f} | Різниця: {val_mae_reg - train_mae_reg:.4f}")
print(f"R2 Score | Train: {train_r2_reg:.4f} | Val: {val_r2_reg:.4f} | Різниця: {train_r2_reg - val_r2_reg:.4f}")


# ==========================================
# ЕТАП 3: Оцінка Всього Пайплайну (Combined Pipeline)
# ==========================================
# Будуємо фінальний прогноз для всієї тренувальної вибірки
train_class_preds = clf.predict(X_train)
train_reg_preds_all = np.expm1(reg.predict(X_train)) # Прогноз регресора для всіх у звичайному просторі
train_pred_final = np.where(train_class_preds == 1, train_reg_preds_all, 0)

# Переводимо в логарифмічний простір для коректного порівняння масштабів
y_train_final_log = np.log1p(y_train_raw)
train_pred_final_log = np.log1p(train_pred_final)

# Метрики пайплайну
train_mae_pipe = mean_absolute_error(y_train_final_log, train_pred_final_log)
# val_mae_pipe беремо з твого попереднього коду (або рахуємо заново)
val_class_preds = clf.predict(X_val)
val_reg_preds_all = np.expm1(reg.predict(X_val))
val_pred_final = np.where(val_class_preds == 1, val_reg_preds_all, 0)

y_val_final_log = np.log1p(y_val_raw)
val_pred_final_log = np.log1p(val_pred_final)

val_mae_pipe = mean_absolute_error(y_val_final_log, val_pred_final_log)
train_r2_pipe = r2_score(y_train_final_log, train_pred_final_log)
val_r2_pipe = r2_score(y_val_final_log, val_pred_final_log)

print(f"\n[Combined Pipeline (All Clients, Log-Space)]")
print(f"MAE      | Train: {train_mae_pipe:.4f} | Val: {val_mae_pipe:.4f} | Різниця: {val_mae_pipe - train_mae_pipe:.4f}")
print(f"R2 Score | Train: {train_r2_pipe:.4f} | Val: {val_r2_pipe:.4f} | Різниця: {train_r2_pipe - val_r2_pipe:.4f}")
print("=" * 60)