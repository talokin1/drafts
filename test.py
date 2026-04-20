from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import numpy as np

print("=" * 60)
print(" ДІАГНОСТИКА У ЗВИЧАЙНОМУ ПРОСТОРІ (РЕАЛЬНІ ГРОШІ)")
print("=" * 60)

# ==========================================
# ЕТАП 2: Регресор (Тільки активні клієнти)
# ==========================================
# Повертаємо цільову змінну та прогнози з лог-простору
y_train_reg_real = np.expm1(y_train_reg_log)
y_val_reg_real = np.expm1(y_val_reg_log)

train_reg_preds_real = np.expm1(train_reg_preds_log)
val_reg_preds_real = np.expm1(val_reg_preds_log)

train_mae_reg_real = mean_absolute_error(y_train_reg_real, train_reg_preds_real)
val_mae_reg_real = mean_absolute_error(y_val_reg_real, val_reg_preds_real)

train_medae_reg_real = median_absolute_error(y_train_reg_real, train_reg_preds_real)
val_medae_reg_real = median_absolute_error(y_val_reg_real, val_reg_preds_real)

train_r2_reg_real = r2_score(y_train_reg_real, train_reg_preds_real)
val_r2_reg_real = r2_score(y_val_reg_real, val_reg_preds_real)

print(f"\n[Stage 2: Regressor (Only Active Clients, ORIGINAL SPACE)]")
print(f"MAE      | Train: {train_mae_reg_real:,.2f} | Val: {val_mae_reg_real:,.2f} | Різниця: {val_mae_reg_real - train_mae_reg_real:,.2f}")
print(f"MedAE    | Train: {train_medae_reg_real:,.2f} | Val: {val_medae_reg_real:,.2f} | (Медіанна похибка)")
print(f"R2 Score | Train: {train_r2_reg_real:.4f} | Val: {val_r2_reg_real:.4f} | (Обережно: чутливо до викидів!)")

# ==========================================
# ЕТАП 3: Весь Пайплайн (Всі клієнти, включно з нулями)
# ==========================================
# y_train_raw та y_val_raw вже у звичайному просторі
# train_pred_final та val_pred_final теж (ми робили expm1 у попередньому скрипті)

train_mae_pipe_real = mean_absolute_error(y_train_raw, train_pred_final)
val_mae_pipe_real = mean_absolute_error(y_val_raw, val_pred_final)

train_medae_pipe_real = median_absolute_error(y_train_raw, train_pred_final)
val_medae_pipe_real = median_absolute_error(y_val_raw, val_pred_final)

train_r2_pipe_real = r2_score(y_train_raw, train_pred_final)
val_r2_pipe_real = r2_score(y_val_raw, val_pred_final)

print(f"\n[Combined Pipeline (All Clients, ORIGINAL SPACE)]")
print(f"MAE      | Train: {train_mae_pipe_real:,.2f} | Val: {val_mae_pipe_real:,.2f} | Різниця: {val_mae_pipe_real - train_mae_pipe_real:,.2f}")
print(f"MedAE    | Train: {train_medae_pipe_real:,.2f} | Val: {val_medae_pipe_real:,.2f} | (Медіанна похибка)")
print(f"R2 Score | Train: {train_r2_pipe_real:.4f} | Val: {val_r2_pipe_real:.4f}")
print("=" * 60)