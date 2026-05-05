import numpy as np
import pandas as pd
import joblib


def get_model_group(segment):
    if str(segment).upper() == "LARGE":
        return "LARGE"
    return "MICRO_SMALL"


def predict_potential_income(
    X_new,
    segment_new=None,
    model_path=r"C:\Projects\(DS-450) Corp potential income\scripts\models\pickle_models\Assets_bucket_ev_model.pkl"
):
    """
    Інференс для Segmented Bucket Expected Value Model.

    Parameters
    ----------
    X_new : pd.DataFrame
        Датафрейм з фічами для моделі.
        Має містити ті самі колонки, що були при навчанні.

    segment_new : pd.Series або None
        Сегмент клієнта: MICRO / SMALL / LARGE.
        Якщо None, функція спробує взяти сегмент із X_new["FIRM_TYPE"].

    model_path : str
        Шлях до збереженої моделі.

    Returns
    -------
    np.array
        Предикти потенційного прибутку.
    """

    artifact = joblib.load(model_path)

    features = artifact["features"]
    cat_cols = artifact["cat_cols"]
    cat_values = artifact["cat_values"]
    groups = artifact["groups"]

    # =========================
    # CHECK FEATURES
    # =========================

    missing_cols = set(features) - set(X_new.columns)

    if missing_cols:
        raise ValueError(f"У нових даних бракує необхідних колонок: {missing_cols}")

    X_new = X_new[features].copy()

    # =========================
    # GET SEGMENTS
    # =========================

    if segment_new is None:
        if "FIRM_TYPE" in X_new.columns:
            segment_new = X_new["FIRM_TYPE"]
        else:
            raise ValueError(
                "segment_new не передано, і в X_new немає колонки FIRM_TYPE"
            )

    segment_new = pd.Series(segment_new, index=X_new.index).astype(str)

    # =========================
    # CATEGORICAL PREP
    # =========================

    for c in cat_cols:
        if c in X_new.columns:
            X_new[c] = pd.Categorical(
                X_new[c],
                categories=cat_values[c]
            )

    # =========================
    # PREDICT
    # =========================

    final_predictions = pd.Series(0.0, index=X_new.index)

    model_groups = segment_new.apply(get_model_group)

    for group_name, pack in groups.items():
        group_mask = model_groups == group_name

        if group_mask.sum() == 0:
            continue

        X_part = X_new.loc[group_mask].copy()
        segment_part = segment_new.loc[group_mask]

        model = pack["model"]
        bucket_medians = pack["bucket_medians"]
        calibration_factors = pack["calibration_factors"]
        segment_caps = pack["segment_caps"]

        proba = model.predict_proba(X_part)
        classes = model.classes_

        pred = np.zeros(len(X_part))

        for i, cls in enumerate(classes):
            pred += proba[:, i] * bucket_medians.get(int(cls), 0.0)

        pred = pd.Series(pred, index=X_part.index)

        # segment-level calibration + caps
        for seg in segment_part.unique():
            seg_mask = segment_part == seg

            factor = calibration_factors.get(seg, 1.0)
            cap = segment_caps.get(seg, np.inf)

            pred.loc[seg_mask] = pred.loc[seg_mask] * factor
            pred.loc[seg_mask] = np.clip(pred.loc[seg_mask], 0, cap)

        final_predictions.loc[group_mask] = pred

    final_predictions = np.clip(final_predictions.values, 0, None)

    return final_predictions


predictions = predict_potential_income(
    X_new=df,
    segment_new=df["FIRM_TYPE"],
    model_path=r"C:\Projects\(DS-450) Corp potential income\scripts\models\pickle_models\Assets_bucket_ev_model.pkl"
)

df_result = df.copy()
df_result["ASSETS_POTENTIAL"] = predictions