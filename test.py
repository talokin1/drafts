from sklearn.metrics import precision_recall_curve

# =========================
# AUTO THRESHOLD — НЕ ЗАНАДТО ЖОРСТКИЙ
# =========================

precision, recall, thresholds = precision_recall_curve(y_val_active, val_p_active)

pr_df = pd.DataFrame({
    "threshold": np.r_[thresholds, 1.0],
    "precision": precision,
    "recall": recall
})

MIN_PRECISION = 0.60
MIN_RECALL = 0.50

candidates = pr_df[
    (pr_df["precision"] >= MIN_PRECISION) &
    (pr_df["recall"] >= MIN_RECALL)
].copy()

if len(candidates) > 0:
    ZERO_THRESHOLD = candidates.sort_values("recall", ascending=False)["threshold"].iloc[0]
else:
    true_active_rate = y_val_active.mean()
    ZERO_THRESHOLD = np.quantile(val_p_active, 1 - true_active_rate)

print("Selected ZERO_THRESHOLD:", ZERO_THRESHOLD)
print("True active rate:", y_val_active.mean())
print("Predicted active rate:", np.mean(val_p_active >= ZERO_THRESHOLD))


# =========================
# BIAS CORRECTION ДО EXPECTED VALUE
# =========================

train_income_if_active = train_income_if_active * bias_correction
val_income_if_active = val_income_if_active * bias_correction


# =========================
# SMOOTH EXPECTED VALUE
# =========================

GAMMA = 1.3

train_prob_factor = train_p_active ** GAMMA
val_prob_factor = val_p_active ** GAMMA

train_expected_raw = train_prob_factor * train_income_if_active
val_expected_raw = val_prob_factor * val_income_if_active


# =========================
# SOFT ZERO ZONE
# =========================
# Не рубаємо все в 0, а плавно зменшуємо слабких клієнтів

SOFT_ZERO_LOW = ZERO_THRESHOLD * 0.5
SOFT_ZERO_HIGH = ZERO_THRESHOLD

def soft_zero_multiplier(p, low, high):
    p = np.asarray(p)
    m = (p - low) / (high - low)
    m = np.clip(m, 0, 1)
    return m

train_soft_mult = soft_zero_multiplier(train_p_active, SOFT_ZERO_LOW, SOFT_ZERO_HIGH)
val_soft_mult = soft_zero_multiplier(val_p_active, SOFT_ZERO_LOW, SOFT_ZERO_HIGH)

train_expected_raw = train_expected_raw * train_soft_mult
val_expected_raw = val_expected_raw * val_soft_mult


# =========================
# ХВІСТ НЕ ЧІПАЄМО АГРЕСИВНО
# =========================

TAIL_BOOST = 0.0

train_expected_raw *= (1 + TAIL_BOOST * (train_p_active > 0.8))
val_expected_raw *= (1 + TAIL_BOOST * (val_p_active > 0.8))