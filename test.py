bad_cols = [
    "CURR_ACC",
    "INCOME_LIABILITIES",
    "LIABILITIES_POTENTIAL",
    "MONTHLY_INCOME",
    "ACCOUNTS_POTENTIAL",
    "Predicted",
    "True_Value"
]

[c for c in bad_cols if c in feature_cols]






df_debug = df.copy()

df_debug["P_ACTIVE"] = p_active
df_debug["INCOME_IF_ACTIVE"] = income_if_active
df_debug["EXPECTED_RAW"] = (p_active ** GAMMA) * income_if_active
df_debug["CALIBRATION_FACTOR"] = tmp["factor"].values
df_debug["EXPECTED_AFTER_CALIBRATION"] = pred
df_debug["CAP_USED"] = caps
df_debug["LIABILITIES_POTENTIAL"] = np.minimum(pred, caps)

debug_cols = [
    "IDENTIFYCODE",
    "FIRM_TYPE",
    "INCOME_LIABILITIES",
    "P_ACTIVE",
    "INCOME_IF_ACTIVE",
    "EXPECTED_RAW",
    "CALIBRATION_FACTOR",
    "EXPECTED_AFTER_CALIBRATION",
    "CAP_USED",
    "LIABILITIES_POTENTIAL"
]

df_debug[debug_cols].sort_values(
    "LIABILITIES_POTENTIAL",
    ascending=False
).head(50)