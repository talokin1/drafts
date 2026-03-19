print("\n--- Training VIP Regressor (ORIGINAL TARGET) ---")
reg_vip = lgb.LGBMRegressor(
    objective="tweedie", # Або "gamma"
    tweedie_variance_power=1.5, # Тюнінг-параметр від 1 до 2 (1 = Poisson, 2 = Gamma)
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=15,
    min_child_samples=10,
    random_state=RANDOM_STATE
)

# ВЧИМО НА ОРИГІНАЛЬНИХ ГРОШАХ (без _log)
reg_vip.fit(
    X_train_vip, np.expm1(y_train_vip), 
    eval_set=[(X_val_vip, np.expm1(y_val_vip_log))],
    eval_metric="l1", # Для Early Stopping можемо залишити MAE
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
)

# Прогноз одразу в грошах
pred_vip_orig = reg_vip.predict(X_val_final)










# 1. Рахуємо залишки на валідаційному сеті VIP у лог-просторі
residuals = y_val_vip_log - reg_vip.predict(X_val_vip)

# 2. Рахуємо дисперсію помилок (MSE)
sigma_squared = np.var(residuals)

# 3. Додаємо дисперсійну поправку при інференсі
pred_vip_log = reg_vip.predict(X_val_final)
pred_vip_orig = np.expm1(pred_vip_log + (sigma_squared / 2))