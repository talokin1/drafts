import numpy as np
import pandas as pd
import joblib


# ============================================================
# 1. LOAD MODEL ARTIFACTS
# ============================================================

MODEL_PATH = r"C:\Projects\DS-450_Corp_potential_income\scripts\models\expected_income_v2_segment_bucket_corrected.pkl"

model = joblib.load(MODEL_PATH)

clf = model["clf"]
reg = model["reg"]

feature_cols = model["feature_cols"]
cat_cols = model["cat_cols"]
cat_values = model["cat_values"]

ACTIVE_THRESHOLD = model["ACTIVE_THRESHOLD"]
CLASSIFICATION_THRESHOLD = model["CLASSIFICATION_THRESHOLD"]
GAMMA = model["GAMMA"]
ZERO_THRESHOLD = model["ZERO_THRESHOLD"]
bias_correction = model["bias_correction"]

SEGMENT_COL = model["SEGMENT_COL"]

calibration_table = model["calibration_table"]
segment_calibration_table = model["segment_calibration_table"]
global_calibration_factor = model["global_calibration_factor"]
pred_decile_edges = model["pred_decile_edges"]

caps_by_segment = model["caps_by_segment"]
global_cap = model["global_cap"]

segment_bucket_correction = model["segment_bucket_correction"]
segment_bucket_fallback = model["segment_bucket_fallback"]
global_segment_bucket_factor = model["global_segment_bucket_factor"]
pred_bucket_edges = model["pred_bucket_edges"]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def apply_fixed_bins(values, bin_edges):
    values = pd.Series(values).clip(lower=0)

    bucket = pd.cut(
        values,
        bins=bin_edges,
        labels=False,
        include_lowest=True
    )

    bucket = pd.Series(bucket).fillna(0).astype(int)

    return bucket.values


def prepare_X_for_inference(df, feature_cols, cat_cols, cat_values):
    X_inf = df.copy()

    missing_cols = [c for c in feature_cols if c not in X_inf.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in inference dataframe: {missing_cols}")

    X_inf = X_inf[feature_cols].copy()

    for c in cat_cols:
        X_inf[c] = pd.Categorical(
            X_inf[c],
            categories=cat_values[c]
        )

    return X_inf


def apply_calibration_inference(
    df_raw,
    pred_raw,
    calibration_table,
    segment_calibration_table,
    global_calibration_factor,
    pred_decile_edges,
    segment_col
):
    tmp = pd.DataFrame(index=df_raw.index)

    if segment_col in df_raw.columns:
        tmp[segment_col] = df_raw[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["pred_raw"] = np.clip(pred_raw, 0, None)
    tmp["pred_decile"] = apply_fixed_bins(tmp["pred_raw"], pred_decile_edges)

    tmp = tmp.merge(
        calibration_table[[segment_col, "pred_decile", "factor"]],
        on=[segment_col, "pred_decile"],
        how="left"
    )

    tmp = tmp.merge(
        segment_calibration_table[[segment_col, "segment_factor"]],
        on=segment_col,
        how="left"
    )

    tmp["factor"] = tmp["factor"].fillna(tmp["segment_factor"])
    tmp["factor"] = tmp["factor"].fillna(global_calibration_factor)
    tmp["factor"] = tmp["factor"].fillna(1.0)

    pred_calibrated = pred_raw * tmp["factor"].values
    pred_calibrated = np.clip(pred_calibrated, 0, None)

    return pred_calibrated, tmp["factor"].values, tmp["pred_decile"].values


def apply_caps_inference(
    df_raw,
    pred,
    caps_by_segment,
    global_cap,
    segment_col
):
    pred = np.array(pred, dtype=float)

    caps = np.full(len(pred), global_cap, dtype=float)

    if segment_col in df_raw.columns:
        segments = df_raw[segment_col].astype(str).values

        for i, seg in enumerate(segments):
            caps[i] = caps_by_segment.get(seg, global_cap)

    pred_capped = np.minimum(pred, caps)
    pred_capped = np.clip(pred_capped, 0, None)

    return pred_capped, caps


def apply_segment_bucket_correction_inference(
    df_raw,
    pred,
    segment_bucket_correction,
    segment_bucket_fallback,
    global_segment_bucket_factor,
    pred_bucket_edges,
    segment_col
):
    pred = np.array(pred, dtype=float)

    tmp = pd.DataFrame(index=df_raw.index)
    tmp["pred"] = np.clip(pred, 0, None)

    if segment_col in df_raw.columns:
        tmp[segment_col] = df_raw[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["segment_pred_bucket"] = apply_fixed_bins(tmp["pred"], pred_bucket_edges)

    tmp = tmp.merge(
        segment_bucket_correction[[segment_col, "pred_bucket", "factor"]],
        left_on=[segment_col, "segment_pred_bucket"],
        right_on=[segment_col, "pred_bucket"],
        how="left"
    )

    tmp = tmp.merge(
        segment_bucket_fallback[[segment_col, "segment_factor"]],
        on=segment_col,
        how="left"
    )

    tmp["factor"] = tmp["factor"].fillna(tmp["segment_factor"])
    tmp["factor"] = tmp["factor"].fillna(global_segment_bucket_factor)
    tmp["factor"] = tmp["factor"].fillna(1.0)

    pred_corrected = pred * tmp["factor"].values
    pred_corrected = np.clip(pred_corrected, 0, None)

    return pred_corrected, tmp["factor"].values, tmp["segment_pred_bucket"].values


# ============================================================
# 3. INFERENCE FUNCTION
# ============================================================

def predict_liabilities_potential(df_new):
    df_raw = df_new.copy()

    # -----------------------------
    # Prepare features
    # -----------------------------
    X_inf = prepare_X_for_inference(
        df=df_raw,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        cat_values=cat_values
    )

    # -----------------------------
    # Stage 1: classifier
    # -----------------------------
    p_active = clf.predict_proba(X_inf)[:, 1]

    # -----------------------------
    # Stage 2: regressor
    # -----------------------------
    pred_log = reg.predict(X_inf)

    income_if_active = np.expm1(pred_log)
    income_if_active = income_if_active * bias_correction
    income_if_active = np.clip(income_if_active, 0, None)

    # -----------------------------
    # Expected value
    # -----------------------------
    expected_raw = (p_active ** GAMMA) * income_if_active

    expected_raw[p_active < ZERO_THRESHOLD] = 0
    expected_raw = np.clip(expected_raw, 0, None)

    # -----------------------------
    # Calibration
    # -----------------------------
    expected_calibrated, calibration_factor, pred_decile = apply_calibration_inference(
        df_raw=df_raw,
        pred_raw=expected_raw,
        calibration_table=calibration_table,
        segment_calibration_table=segment_calibration_table,
        global_calibration_factor=global_calibration_factor,
        pred_decile_edges=pred_decile_edges,
        segment_col=SEGMENT_COL
    )

    # -----------------------------
    # Caps
    # -----------------------------
    pred_capped, cap_used = apply_caps_inference(
        df_raw=df_raw,
        pred=expected_calibrated,
        caps_by_segment=caps_by_segment,
        global_cap=global_cap,
        segment_col=SEGMENT_COL
    )

    # -----------------------------
    # Segment × prediction bucket correction
    # -----------------------------
    final_pred, segment_bucket_factor, segment_pred_bucket = apply_segment_bucket_correction_inference(
        df_raw=df_raw,
        pred=pred_capped,
        segment_bucket_correction=segment_bucket_correction,
        segment_bucket_fallback=segment_bucket_fallback,
        global_segment_bucket_factor=global_segment_bucket_factor,
        pred_bucket_edges=pred_bucket_edges,
        segment_col=SEGMENT_COL
    )

    # -----------------------------
    # Output
    # -----------------------------
    result = df_raw.copy()

    result["P_LIABILITIES_ACTIVE"] = p_active
    result["IS_LIKELY_ACTIVE"] = (p_active >= CLASSIFICATION_THRESHOLD).astype(int)

    result["LIABILITIES_IF_ACTIVE"] = income_if_active
    result["LIABILITIES_EXPECTED_RAW"] = expected_raw

    result["CALIBRATION_FACTOR"] = calibration_factor
    result["PRED_DECILE"] = pred_decile
    result["LIABILITIES_EXPECTED_CALIBRATED"] = expected_calibrated

    result["CAP_USED"] = cap_used
    result["LIABILITIES_AFTER_CAP"] = pred_capped

    result["SEGMENT_BUCKET_FACTOR"] = segment_bucket_factor
    result["SEGMENT_PRED_BUCKET"] = segment_pred_bucket

    result["LIABILITIES_POTENTIAL"] = final_pred

    return result


# ============================================================
# 4. RUN INFERENCE
# ============================================================

# df — твій inference dataframe
df_pred = predict_liabilities_potential(df)


# ============================================================
# 5. OPTIONAL: ROUND OUTPUT
# ============================================================

round_cols = [
    "P_LIABILITIES_ACTIVE",
    "LIABILITIES_IF_ACTIVE",
    "LIABILITIES_EXPECTED_RAW",
    "CALIBRATION_FACTOR",
    "LIABILITIES_EXPECTED_CALIBRATED",
    "CAP_USED",
    "LIABILITIES_AFTER_CAP",
    "SEGMENT_BUCKET_FACTOR",
    "LIABILITIES_POTENTIAL"
]

for c in round_cols:
    if c in df_pred.columns:
        if c in ["P_LIABILITIES_ACTIVE", "CALIBRATION_FACTOR", "SEGMENT_BUCKET_FACTOR"]:
            df_pred[c] = df_pred[c].round(4)
        else:
            df_pred[c] = df_pred[c].round(2)


# ============================================================
# 6. VIEW MAIN RESULT
# ============================================================

main_cols = []

for c in ["IDENTIFYCODE", "FIRM_TYPE"]:
    if c in df_pred.columns:
        main_cols.append(c)

main_cols += [
    "P_LIABILITIES_ACTIVE",
    "LIABILITIES_IF_ACTIVE",
    "LIABILITIES_EXPECTED_RAW",
    "LIABILITIES_EXPECTED_CALIBRATED",
    "LIABILITIES_AFTER_CAP",
    "SEGMENT_BUCKET_FACTOR",
    "LIABILITIES_POTENTIAL"
]

df_result = df_pred[main_cols].copy()
df["LIABILITIES_POTENTIAL"] = df_pred["LIABILITIES_POTENTIAL"].values