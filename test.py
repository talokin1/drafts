import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# 1. Визначаємо поріг "VIP клієнта"
# Давайте глянемо медіану або 75-й перцентиль, щоб виділити топ компаній
threshold = df['CURR_ACC'].quantile(0.75) 
print(f"Поріг VIP-клієнта: {threshold:.2f} грн")

# 2. Створюємо новий таргет для класифікації
y_class = (df['CURR_ACC'] >= threshold).astype(int)

# 3. Спліт (використовуємо ті самі X, що і раніше)
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# 4. Тренуємо Класифікатор
clf = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced' # Важливо! Щоб звернути увагу на менший клас VIP
)

print("Тренуємо класифікатор...")
clf.fit(X_train, y_train_cls, categorical_feature=cat_features)

# 5. Перевіряємо якість
preds_proba = clf.predict_proba(X_test)[:, 1]
preds_class = clf.predict(X_test)

auc = roc_auc_score(y_test_cls, preds_proba)
print(f"\n--- Результати Класифікації ---")
print(f"ROC-AUC: {auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_cls, preds_class))
print("\nReport:")
print(classification_report(y_test_cls, preds_class))

# 6. Важливість фічей для класифікації
lgb.plot_importance(clf, max_num_features=15, importance_type='gain', title='Top Features for Classifying VIPs')