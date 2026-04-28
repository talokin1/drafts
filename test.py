import joblib

model_artifacts = {
    "clf": clf,
    "reg": reg,
    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "cat_values": cat_values,
    "calibration_table": calibration_table,
    "segment_calibration_table": segment_calibration_table,
    "global_calibration_factor": global_calibration_factor,
    "caps_by_segment": caps_by_segment,
    "global_cap": global_cap,
    "SEGMENT_COL": SEGMENT_COL,
    "N_DECILES": N_DECILES,

    "GAMMA": 2.0,
    "ZERO_THRESHOLD": 0.25,
    "bias_correction": bias_correction
}

joblib.dump(model_artifacts, "expected_income_v1.pkl")

print("Saved")



import joblib
import numpy as np
import pandas as pd

# ========= LOAD =========
model = joblib.load("expected_income_v1.pkl")

clf = model["clf"]
reg = model["reg"]

feature_cols = model["feature_cols"]
cat_cols = model["cat_cols"]
cat_values = model["cat_values"]

calibration_table = model["calibration_table"]
segment_calibration_table = model["segment_calibration_table"]
global_calibration_factor = model["global_calibration_factor"]

caps_by_segment = model["caps_by_segment"]
global_cap = model["global_cap"]

SEGMENT_COL = model["SEGMENT_COL"]
N_DECILES = model["N_DECILES"]

GAMMA = model["GAMMA"]
ZERO_THRESHOLD = model["ZERO_THRESHOLD"]
bias_correction = model["bias_correction"]

X = df.copy()
X = X[feature_cols]

for c in cat_cols:
    X[c] = pd.Categorical(X[c], categories=cat_values[c])

p_active = clf.predict_proba(X)[:, 1]
pred_log = reg.predict(X)

income_if_active = np.expm1(pred_log)
income_if_active = income_if_active * bias_correction
income_if_active = np.clip(income_if_active, 0, None)

pred = (p_active ** GAMMA) * income_if_active

# zero correction
pred[p_active < ZERO_THRESHOLD] = 0

pred_series = pd.Series(pred)

try:
    deciles = pd.qcut(pred_series, q=N_DECILES, labels=False, duplicates="drop")
except:
    deciles = pd.Series(np.zeros(len(pred)), index=pred_series.index)


tmp = pd.DataFrame({
    "pred": pred,
    "decile": deciles
})

if SEGMENT_COL in df.columns:
    tmp[SEGMENT_COL] = df[SEGMENT_COL].astype(str).values
else:
    tmp[SEGMENT_COL] = "ALL"


tmp = tmp.merge(
    calibration_table[[SEGMENT_COL, "pred_decile", "factor"]],
    left_on=[SEGMENT_COL, "decile"],
    right_on=[SEGMENT_COL, "pred_decile"],
    how="left"
)

tmp = tmp.merge(
    segment_calibration_table[[SEGMENT_COL, "factor"]].rename(columns={"factor": "segment_factor"}),
    on=SEGMENT_COL,
    how="left"
)

tmp["factor"] = tmp["factor"].fillna(tmp["segment_factor"])
tmp["factor"] = tmp["factor"].fillna(global_calibration_factor)
tmp["factor"] = tmp["factor"].fillna(1.0)

pred = pred * tmp["factor"].values

df["LIABILITIES_POTENTIAL"] = pred