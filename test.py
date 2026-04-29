from sklearn.isotonic import IsotonicRegression

# беремо тільки ненульові predicted, бо нулі окремо контролює classifier
mask_cal = val_expected_raw > ACTIVE_THRESHOLD

pred_log_cal = np.log1p(val_expected_raw[mask_cal])
true_log_cal = np.log1p(y_val_raw.values[mask_cal])

# сортуємо pred і мапимо його квантілі на true distribution
order = np.argsort(pred_log_cal)
pred_sorted = pred_log_cal[order]
true_sorted = np.sort(true_log_cal)

quantile_mapper = IsotonicRegression(out_of_bounds="clip")

quantile_mapper.fit(
    pred_sorted,
    true_sorted
)

def apply_quantile_mapping(pred):
    pred = np.asarray(pred, dtype=float)
    out = pred.copy()

    mask = pred > ACTIVE_THRESHOLD
    out[mask] = np.expm1(
        quantile_mapper.predict(np.log1p(pred[mask]))
    )

    out = np.clip(out, 0, None)
    return out

train_expected_raw = apply_quantile_mapping(train_expected_raw)
val_expected_raw = apply_quantile_mapping(val_expected_raw)