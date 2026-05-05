def make_bucket_values(y_train, bucket_train):
    """
    Консервативне значення bucket-а.
    Для positive bucket беремо не median, а 25-й перцентиль.
    Це зменшує завищення.
    """
    values = {}

    all_buckets = sorted(bucket_train.unique())

    for b in all_buckets:
        if b == 0:
            values[int(b)] = 0.0
        else:
            vals = y_train[bucket_train == b]
            if len(vals) == 0:
                values[int(b)] = 0.0
            else:
                values[int(b)] = float(np.quantile(vals, 0.25))

    return values


bucket_medians = make_bucket_values(y_train, bucket_train)












pred_raw = np.zeros(len(X_part))
positive_proba = np.zeros(len(X_part))

for i, cls in enumerate(classes):
    cls_int = int(cls)

    if cls_int != 0:
        positive_proba += proba[:, i]

    pred_raw += proba[:, i] * medians.get(cls_int, 0.0)

# якщо модель не впевнена, що клієнт активний — ставимо 0
pred_raw = np.where(
    positive_proba >= MIN_POSITIVE_PROBA,
    pred_raw,
    0
)

pred_raw = pd.Series(pred_raw, index=X_part.index)
















def predict_bucket_model(model_pack, X_new, segment_new, min_positive_proba=0.35):
    X_new = X_new.copy()
    segment_new = pd.Series(segment_new, index=X_new.index).astype(str)

    preds = pd.Series(0.0, index=X_new.index)

    for group_name, pack in model_pack["groups"].items():
        group_mask = segment_new.apply(get_model_group) == group_name

        if group_mask.sum() == 0:
            continue

        X_part = X_new.loc[group_mask, model_pack["features"]].copy()

        for c in model_pack["cat_cols"]:
            X_part[c] = pd.Categorical(
                X_part[c],
                categories=model_pack["cat_values"][c]
            )

        model = pack["model"]
        bucket_values = pack["bucket_medians"]

        proba = model.predict_proba(X_part)
        classes = model.classes_

        pred_raw = np.zeros(len(X_part))
        positive_proba = np.zeros(len(X_part))

        for i, cls in enumerate(classes):
            cls_int = int(cls)

            if cls_int != 0:
                positive_proba += proba[:, i]

            pred_raw += proba[:, i] * bucket_values.get(cls_int, 0.0)

        # zero-gate
        pred_raw = np.where(
            positive_proba >= min_positive_proba,
            pred_raw,
            0
        )

        pred_raw = pd.Series(pred_raw, index=X_part.index)

        # segment-level calibration + cap
        for seg in segment_new.loc[group_mask].unique():
            seg_mask = segment_new.loc[group_mask] == seg

            factor = pack["calibration_factors"].get(seg, 1.0)
            cap = pack["segment_caps"].get(seg, np.inf)

            pred_raw.loc[seg_mask] = pred_raw.loc[seg_mask] * factor
            pred_raw.loc[seg_mask] = np.clip(pred_raw.loc[seg_mask], 0, cap)

        preds.loc[group_mask] = pred_raw

    return preds.values




predictions = predict_potential_income(
    X_new=df,
    segment_new=df["FIRM_TYPE"],
    model_path=r"C:\Projects\(DS-450) Corp potential income\scripts\models\pickle_models\Assets_bucket_ev_model.pkl",
    min_positive_proba=0.35
)

df["ASSETS_POTENTIAL"] = predictions



min_positive_proba=0.35











pred = np.zeros(len(X_part))
positive_proba = np.zeros(len(X_part))

for i, cls in enumerate(classes):
    cls_int = int(cls)

    if cls_int != 0:
        positive_proba += proba[:, i]

    pred += proba[:, i] * bucket_medians.get(cls_int, 0.0)

pred = np.where(
    positive_proba >= min_positive_proba,
    pred,
    0
)











