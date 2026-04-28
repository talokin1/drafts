P_ZERO_THRESHOLD = artifacts.get("p_zero_threshold", 0.65)

p_zero = proba[:, 0]
expected_raw[p_zero >= P_ZERO_THRESHOLD] = 0



result["P_ZERO"] = p_zero
result["IS_PRED_ZERO"] = (p_zero >= P_ZERO_THRESHOLD).astype(int)