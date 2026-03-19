# Оцінка Router'а
from sklearn.metrics import roc_auc_score, accuracy_score
print(f"Router ROC-AUC: {roc_auc_score(y_val_class, p_vip_val):.4f}")

# MAE тільки для Mass сегмента (де в реальності клієнт був Mass)
mask_real_mass = y_val_original <= np.expm1(THRESHOLD_LOG)
mae_mass = mean_absolute_error(
    y_val_original[mask_real_mass], 
    pred_final_original[mask_real_mass]
)
print(f"MAE on REAL MASS clients : {mae_mass:,.2f}")

# MAE тільки для VIP сегмента
mask_real_vip = y_val_original > np.expm1(THRESHOLD_LOG)
mae_vip = mean_absolute_error(
    y_val_original[mask_real_vip], 
    pred_final_original[mask_real_vip]
)
print(f"MAE on REAL VIP clients  : {mae_vip:,.2f}")