train_residuals = y_train_reg_log - reg.predict(X_train_reg)
sigma = np.std(train_residuals)

bias_correction = np.exp(0.5 * sigma**2)

print("Log-bias correction factor:", bias_correction)



GAMMA = 2.0          # 1.5–3.0 тюнити
ZERO_THRESHOLD = 0.2 # 0.15–0.3 тюнити

# bias correction
train_income_if_active = train_income_if_active * bias_correction
val_income_if_active = val_income_if_active * bias_correction

# tempered expected value
train_expected_raw = (train_p_active ** GAMMA) * train_income_if_active
val_expected_raw = (val_p_active ** GAMMA) * val_income_if_active

# explicit zero correction
train_expected_raw[train_p_active < ZERO_THRESHOLD] = 0
val_expected_raw[val_p_active < ZERO_THRESHOLD] = 0