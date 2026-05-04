
prob_val = clf.predict_proba(X_val)[:, 1]

val_reg_preds_log = reg.predict(X_val)
val_reg_preds = np.expm1(val_reg_preds_log)
val_reg_preds = np.clip(val_reg_preds, 0, None)


train_reg_preds = np.expm1(reg.predict(X_train_reg))
train_reg_preds = np.clip(train_reg_preds, 0, None)

CAP_Q = 0.99
global_cap = np.quantile(y_train_raw[y_train_raw >= THRESHOLD], CAP_Q)

val_reg_preds_capped = np.clip(val_reg_preds, 0, global_cap)

def evaluate_threshold(y_true, prob, reg_pred, threshold):
    y_pred = np.where(prob >= threshold, reg_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    true_total = y_true.sum()
    pred_total = y_pred.sum()

    total_ratio = pred_total / true_total if true_total > 0 else np.nan

    y_true_active = (y_true >= THRESHOLD).astype(int)
    y_pred_active = (y_pred > 0).astype(int)

    false_positive_rate = np.mean((y_true_active == 0) & (y_pred_active == 1))
    predicted_active_rate = y_pred_active.mean()
    real_active_rate = y_true_active.mean()

    return {
        "threshold": threshold,
        "mae": mae,
        "medae": medae,
        "r2": r2,
        "true_total": true_total,
        "pred_total": pred_total,
        "total_ratio": total_ratio,
        "false_positive_rate": false_positive_rate,
        "predicted_active_rate": predicted_active_rate,
        "real_active_rate": real_active_rate
    }


threshold_results = []

for th in np.arange(0.10, 0.91, 0.01):
    threshold_results.append(
        evaluate_threshold(
            y_true=y_val_raw,
            prob=prob_val,
            reg_pred=val_reg_preds_capped,
            threshold=th
        )
    )

threshold_results = pd.DataFrame(threshold_results)


threshold_results["score"] = (
    np.abs(threshold_results["total_ratio"] - 1.0) * 2.0
    + threshold_results["false_positive_rate"] * 5.0
    + threshold_results["mae"] / threshold_results["mae"].median()
)

best_row = threshold_results.sort_values("score").iloc[0]
BEST_THRESHOLD = best_row["threshold"]

print("=" * 70)
print("BEST CLASSIFICATION THRESHOLD")
print("=" * 70)
print(best_row)

y_pred_final = np.where(prob_val >= BEST_THRESHOLD, val_reg_preds_capped, 0)


y_val_final_log = np.log1p(y_val_raw)
y_pred_final_log = np.log1p(y_pred_final)

mae = mean_absolute_error(y_val_raw, y_pred_final)
medae = median_absolute_error(y_val_raw, y_pred_final)
r2 = r2_score(y_val_raw, y_pred_final)

mae_log = mean_absolute_error(y_val_final_log, y_pred_final_log)
medae_log = median_absolute_error(y_val_final_log, y_pred_final_log)
r2_log = r2_score(y_val_final_log, y_pred_final_log)

print("=" * 70)
print("FINAL COMBINED PIPELINE METRICS")
print("=" * 70)
print(f"Threshold      : {BEST_THRESHOLD:.2f}")
print(f"MAE            : {mae:,.2f}")
print(f"MedAE          : {medae:,.2f}")
print(f"R2             : {r2:.4f}")
print(f"MAE_log        : {mae_log:.5f}")
print(f"MedAE_log      : {medae_log:.5f}")
print(f"R2_log         : {r2_log:.5f}")
print("-" * 70)
print(f"Real total     : {y_val_raw.sum():,.2f}")
print(f"Pred total     : {y_pred_final.sum():,.2f}")
print(f"Pred / Real    : {y_pred_final.sum() / y_val_raw.sum():.4f}")
print("-" * 70)
print(f"Real active rate      : {(y_val_raw >= THRESHOLD).mean():.4f}")
print(f"Predicted active rate : {(y_pred_final > 0).mean():.4f}")
print("=" * 70)