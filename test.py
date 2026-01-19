import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# --- НАЛАШТУВАННЯ ---
THRESHOLD = 1000  # Поріг "VIP-клієнта" (з вашого аналізу)
TARGET_COL = 'CURR_ACC'

# 1. ПІДГОТОВКА ДАНИХ
print("1. Підготовка даних...")
# Прибираємо технічні мінуси (овердрафти стають 0)
df[TARGET_COL] = df[TARGET_COL].clip(lower=0)

# Визначаємо категоріальні фічі
cat_features = [c for c in df.columns if df[c].dtype.name == 'category' or df[c].dtype.name == 'object']
for c in cat_features:
    df[c] = df[c].astype('category')

# X та y
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Створюємо бінарний таргет для стратифікації
y_class = (y > THRESHOLD).astype(int)

# Спліт на Train/Test (80/20)
X_train, X_test, y_train, y_test, y_cls_train, y_cls_test = train_test_split(
    X, y, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# --- ЕТАП 1: ВІДБІР ФІЧ (FEATURE SELECTION) ---
print("\n2. Починаємо відбір корисних фічів для Регресії...")

# Готуємо дані ТІЛЬКИ для VIP-сегменту (для відбору фіч)
mask_vip_train = y_cls_train == 1
mask_vip_test = y_cls_test == 1 # Використовуємо тест для чесної перевірки важливості

X_train_vip = X_train[mask_vip_train]
y_train_vip_log = np.log1p(y_train[mask_vip_train])

X_test_vip = X_test[mask_vip_test]
y_test_vip_log = np.log1p(y_test[mask_vip_test])

# Тимчасовий регресор для перевірки фіч
temp_reg = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
temp_reg.fit(X_train_vip, y_train_vip_log, categorical_feature=cat_features)

# Рахуємо Permutation Importance на тестовій вибірці (щоб уникнути оверфіту)
print("   Рахуємо Permutation Importance (це може зайняти хвилину)...")
perm_result = permutation_importance(
    temp_reg, X_test_vip, y_test_vip_log,
    n_repeats=5, random_state=42, n_jobs=-1
)

# Фільтруємо: залишаємо тільки ті, де важливість > 0
importance_df = pd.DataFrame({'feature': X.columns, 'importance': perm_result.importances_mean})
USEFUL_FEATURES = importance_df[importance_df['importance'] > 0]['feature'].tolist()

print(f"   Знайдено корисних фіч: {len(USEFUL_FEATURES)} із {len(X.columns)}")
print(f"   ТОП-5 фіч: {USEFUL_FEATURES[:5]}")

if len(USEFUL_FEATURES) == 0:
    print("   УВАГА: Не знайдено жодної корисної фічі! Використовуємо всі (але це погано).")
    USEFUL_FEATURES = X.columns.tolist()

# Оновлюємо список категоріальних фіч (тільки ті, що вижили)
cat_features_opt = [c for c in USEFUL_FEATURES if c in cat_features]


# --- ЕТАП 2: НАВЧАННЯ ФІНАЛЬНИХ МОДЕЛЕЙ ---
print("\n3. Навчання фінальних моделей...")

# А. Класифікатор (Використовуємо ВСІ фічі, бо класифікатор стійкий до шуму)
print("   Навчаємо Classifier (Prob > 1000 грн)...")
clf = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    class_weight='balanced', random_state=42, verbose=-1
)
clf.fit(X_train, y_cls_train, categorical_feature=cat_features)

# Б. Регресор (ТІЛЬКИ на корисних фічах!)
print("   Навчаємо Regressor (Amount)...")
reg = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    random_state=42, verbose=-1
)
# Навчаємо на чистих фічах
reg.fit(
    X_train_vip[USEFUL_FEATURES], 
    y_train_vip_log, 
    categorical_feature=cat_features_opt
)


# --- ЕТАП 3: ФІНАЛЬНИЙ ПРОГНОЗ ТА ОЦІНКА ---
print("\n4. Оцінка результатів (Two-Stage Model)...")

def predict_two_stage(X_input, classifier, regressor, features_reg):
    # 1. Ймовірність (Probability)
    probs = classifier.predict_proba(X_input)[:, 1]
    
    # 2. Сума (Prediction) - подаємо тільки корисні фічі
    pred_log = regressor.predict(X_input[features_reg])
    pred_amount = np.expm1(pred_log)
    
    # 3. Soft Gating (Ймовірність * Сума)
    return probs * pred_amount

# Прогноз на тесті
final_preds = predict_two_stage(X_test, clf, reg, USEFUL_FEATURES)

# Метрики
mae = mean_absolute_error(y_test, final_preds)
r2 = r2_score(y_test, final_preds)
auc = roc_auc_score(y_cls_test, clf.predict_proba(X_test)[:, 1])

print("-" * 30)
print(f"ROC-AUC (Класифікація): {auc:.4f}")
print(f"Final MAE: {mae:.2f} грн")
print(f"Final R2:  {r2:.4f}")
print("-" * 30)

# Візуалізація: Реальні vs Прогноз
plt.figure(figsize=(10, 6))
# Малюємо тільки точки > 0, щоб логарифмічна шкала не ламалася
mask_plot = (y_test > 0) & (final_preds > 0)
plt.scatter(y_test[mask_plot], final_preds[mask_plot], alpha=0.3, s=10)
plt.plot([1, y_test.max()], [1, y_test.max()], 'r--', lw=2) # Лінія ідеалу
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Реальна сума (Log)')
plt.ylabel('Прогноз моделі (Log)')
plt.title('Фінальний результат: True vs Predicted')
plt.show()