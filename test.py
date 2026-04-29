QM_BLEND = 0.65  # 0.5–0.8 тюнити

def apply_quantile_mapping(pred):
    pred = np.asarray(pred, dtype=float)
    out = pred.copy()

    mask = pred > ACTIVE_THRESHOLD

    mapped = np.expm1(
        quantile_mapper.predict(np.log1p(pred[mask]))
    )

    # не повністю замінюємо pred, а змішуємо
    out[mask] = (1 - QM_BLEND) * pred[mask] + QM_BLEND * mapped

    out = np.clip(out, 0, None)
    return out