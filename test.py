y_val_raw   = np.expm1(y_val)      # true грн
y_pred_raw  = np.expm1(y_pred)     # pred грн

mae_uah = mean_absolute_error(y_val_raw, y_pred_raw)
print("MAE (UAH):", mae_uah)


mae_zero_uah = mean_absolute_error(y_val_raw, np.zeros_like(y_val_raw))

med_log = np.median(y_train)
med_uah = np.expm1(med_log)
pred_med_uah = np.full_like(y_val_raw, med_uah)

mae_med_uah = mean_absolute_error(y_val_raw, pred_med_uah)

print("Baseline ZERO MAE (UAH):", mae_zero_uah)
print("Baseline MEDIAN MAE (UAH):", mae_med_uah)
