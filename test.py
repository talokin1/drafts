# =========================
# SEGMENT-AWARE EXPECTED VALUE
# =========================

GAMMA_BY_SEGMENT = {
    "MICRO": 5.0,
    "SMALL": 4.0,
    "MEDIUM": 3.0,
    "LARGE": 2.0
}

ZERO_THRESHOLD_BY_SEGMENT = {
    "MICRO": 0.55,
    "SMALL": 0.45,
    "MEDIUM": 0.35,
    "LARGE": 0.25
}

MANUAL_CAPS_BY_SEGMENT = {
    "MICRO": 2_000,
    "SMALL": 7_000,
    "MEDIUM": 25_000,
    "LARGE": 150_000
}

segments = df["FIRM_TYPE"].astype(str).values

gamma_arr = np.array([
    GAMMA_BY_SEGMENT.get(seg, 3.0)
    for seg in segments
])

zero_threshold_arr = np.array([
    ZERO_THRESHOLD_BY_SEGMENT.get(seg, 0.4)
    for seg in segments
])

manual_caps = np.array([
    MANUAL_CAPS_BY_SEGMENT.get(seg, 50_000)
    for seg in segments
])

# base expected value
pred = (p_active ** gamma_arr) * income_if_active

# explicit zero correction
pred[p_active < zero_threshold_arr] = 0

# manual business caps
pred = np.minimum(pred, manual_caps)

# safety
pred = np.clip(pred, 0, None)

df["LIABILITIES_POTENTIAL"] = pred