# 1. Створюємо і чистимо X_inf ОДИН РАЗ
expected_features = model.feature_names_
X_inf = df_inf[expected_features].copy()

cat_features = X_inf.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
X_inf[cat_features] = X_inf[cat_features].fillna('Missing').astype(str)

INFERENCE_THRESHOLD = 0.418

from sklearn.calibration import CalibratedClassifierCV

# 2. Створюємо та фітимо калібратор
calibrated_model = CalibratedClassifierCV(
    estimator=model,
    method='isotonic',
    cv='prefit'
)

calibrated_model.fit(X_val, y_val)

# ЗАВЖДИ ПЕРЕВІРЯЙ: X_val теж не повинен містити NaN у категоріальних фічах. 
# Якщо fit() пройшов успішно, значить з X_val все ок.

# 3. Інференс (використовуємо наш підготовлений X_inf, не перезаписуємо його!)
df_inf['hnwi_prob'] = calibrated_model.predict_proba(X_inf)[:, 1]

df_inf['is_hnwi_car'] = (df_inf['hnwi_prob'] >= INFERENCE_THRESHOLD).astype(int)

print(f"Проскороно автомобілів: {len(df_inf)}")
print(f"Знайдено HNWI-автомобілів: {df_inf['is_hnwi_car'].sum()}")