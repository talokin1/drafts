import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# --- 1. Попередня обробка та Feature Engineering ---
print("Підготовка даних...")

# Логарифмування вхідних числових фічей (Дуже важливо для фінансів!)
# Шукаємо стовпчики, схожі на гроші (можна задати список вручну)
money_cols = ['REVENUE_CUR', 'REVENUE_PREV', 'CASH_CURR', 'NB_EMPL'] # Додайте свої
for col in money_cols:
    if col in df.columns:
        # log1p(x) = log(x + 1) - безпечно для нулів
        df[f'{col}_LOG'] = np.log1p(df[col].clip(lower=0)) 

# Створення нових фічей (Ratios)
if 'CASH_CURR' in df.columns and 'REVENUE_CUR' in df.columns:
    # Яка частка кешу від доходу?
    df['CASH_TO_REVENUE'] = df['CASH_CURR'] / (df['REVENUE_CUR'] + 1)

# Категоріальні змінні
cat_features = [c for c in df.columns if df[c].dtype.name in ['category', 'object']]
for c in cat_features:
    df[c] = df[c].astype('category')

# --- 2. Розбиття даних (Correct Split) ---
THRESHOLD = 1000
TARGET_COL = 'CURR_ACC'

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].clip(lower=0)

# Створюємо бінарний таргет для стратифікації
y_class = (y > THRESHOLD).astype(int)

# 1. Відрізаємо Тест (Holdout) - його НЕ ЧІПАЄМО до фінального звіту
X_temp, X_test, y_temp, y_test, y_cls_temp, y_cls_test = train_test_split(
    X, y, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# 2. Розбиваємо залишок на Train та Validation (для відбору фічей та тюнінгу)
X_train, X_val, y_train, y_val, y_cls_train, y_cls_val = train_test_split(
    X_temp, y_temp, y_cls_temp, test_size=0.25, random_state=42, stratify=y_cls_temp
) 
# (0.25 від 80% = 20% загальної вибірки. Разом: 60% Train, 20% Val, 20% Test)

# --- 3. Підготовка для Регресії (Log Target) ---
# Навчаємо регресію тільки на "платниках" (y > 0 або > THRESHOLD)
mask_vip_train = y_cls_train == 1
mask_vip_val   = y_cls_val == 1

X_train_reg = X_train[mask_vip_train]
y_train_reg_log = np.log1p(y_train[mask_vip_train])

X_val_reg = X_val[mask_vip_val]
y_val_reg_log = np.log1p(y_val[mask_vip_val])

# --- 4. Відбір фічей (на Validation сеті!) ---
print("\nВідбір корисних фічей (Permutation Importance)...")

temp_reg = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
temp_reg.fit(X_train_reg, y_train_reg_log, categorical_feature=cat_features)

# ВАЖЛИВО: Рахуємо важливість на X_val, а не на X_test!
perm_result = permutation_importance(
    temp_reg, X_val_reg, y_val_reg_log,
    n_repeats=5, random_state=42, n_jobs=-1
)

importance_df = pd.DataFrame({'feature': X.columns, 'importance': perm_result.importances_mean})
# Беремо тільки ті, що реально покращують метрику (> 0)
USEFUL_FEATURES = importance_df[importance_df['importance'] > 0]['feature'].tolist()

print(f"Знайдено корисних фічей: {len(USEFUL_FEATURES)} із {len(X.columns)}")
print(f"ТОП-5: {USEFUL_FEATURES[:5]}")

# Якщо список порожній (рідко, але буває)
if not USEFUL_FEATURES:
    USEFUL_FEATURES = X.columns.tolist()

# Оновлюємо список категоріальних фічей (тільки ті, що лишилися)
cat_features_opt = [c for c in USEFUL_FEATURES if c in cat_features]

# --- 5. Навчання Фінальних Моделей ---
print("\nНавчання фінальних моделей...")

# Класифікатор (можна на всіх фічах, або теж на USEFUL)
clf = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    class_weight='balanced', random_state=42, verbose=-1
)
clf.fit(X_train, y_cls_train, categorical_feature=cat_features, 
        eval_set=[(X_val, y_cls_val)], callbacks=[lgb.early_stopping(50)])

# Регресор (Тільки на корисних фічах)
reg = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    objective='regression_l1', # Спробуйте L1 (MAE) або 'tweedie'
    random_state=42, verbose=-1
)
reg.fit(
    X_train_reg[USEFUL_FEATURES], 
    y_train_reg_log,
    categorical_feature=cat_features_opt,
    eval_set=[(X_val_reg[USEFUL_FEATURES], y_val_reg_log)],
    callbacks=[lgb.early_stopping(50)]
)

print("\nГотово! Тепер можна перевіряти на X_test.")