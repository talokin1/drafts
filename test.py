from sklearn.calibration import CalibratedClassifierCV

# 1. Твоя модель вже навчена (наприклад, після твого циклу або просто model.fit)
# model = CatBoostClassifier(...)
# model.fit(train_pool, ...)

# 2. Створюємо калібратор. 
# cv='prefit' гарантує, що CatBoost НЕ буде перенавчатися.
# method='isotonic' найкраще підходить для дерев та великих вибірок.
calibrated_model = CalibratedClassifierCV(
    estimator=model, 
    method='isotonic', 
    cv='prefit' 
)

# 3. "Навчаємо" тільки саму шкалу калібрування на ВАЛІДАЦІЙНИХ даних!
# Це відбувається миттєво, бо це просто підбір однієї функції.
calibrated_model.fit(X_val, y_val)

# 4. Тепер робимо інференс на нових даних
expected_features = model.feature_names_
X_inf = df_inf[expected_features].copy()

# Використовуємо відкалібровану модель замість оригінальної
# Тепер ці ймовірності будуть математично чесними від 0.0 до 1.0
df_inf['hnwi_prob_calibrated'] = calibrated_model.predict_proba(X_inf)[:, 1]