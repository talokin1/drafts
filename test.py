results = []

for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
    val_is_profitable = (y_pred_proba >= thr)

    preds = np.zeros(len(X_val_clf))

    if val_is_profitable.sum() > 0:
        preds[val_is_profitable] = reg.predict(X_val_clf[val_is_profitable])

    preds_final = np.expm1(preds)
    y_true = df_val[TARGET_NAME].values

    mae_total = mean_absolute_error(y_true, preds_final)

    # тільки для прибуткових
    mask = y_true > 0
    mae_prof = mean_absolute_error(y_true[mask], preds_final[mask])

    results.append({
        "threshold": thr,
        "mae_total": mae_total,
        "mae_profitable": mae_prof,
        "selected_clients": val_is_profitable.sum()
    })

pd.DataFrame(results).sort_values("mae_total")