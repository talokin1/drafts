import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Припускаємо, що df - це твій датафрейм, де вже видалені нулі (Y > 0)
# TARGET_NAME - назва цільової змінної
# cat_features - список категоріальних фічей

THRESHOLD = 1200
RANDOM_STATE = 42

# --- КРОК 0: Підготовка даних ---
# Створюємо бінарний таргет для маршрутизатора
df['is_high_income'] = (df[TARGET_NAME] > THRESHOLD).astype(int)

# Відділяємо фічі
X = df.drop(columns=[TARGET_NAME, 'is_high_income'])
y_raw = df[TARGET_NAME]
y_router = df['is_high_income']

# Розбиття на train/val з урахуванням пропорції "багатих" клієнтів
X_train, X_val, y_train_raw, y_val_raw, y_train_router, y_val_router = train_test_split(
    X, y_raw, y_router, test_size=0.2, random_state=RANDOM_STATE, stratify=y_router
)

# Для категоріальних фічей LightGBM
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")


# --- КРОК 1: Модель-маршрутизатор (Чи буде дохід > 1200?) ---
print("Навчання маршрутизатора...")
router_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
router_model.fit(
    X_train, y_train_router,
    eval_set=[(X_val, y_val_router)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)


# --- КРОК 2: Гілка "Стандарт" (Y <= 1200) ---
print("\nНавчання моделі для стандартних тарифів...")
# Фільтруємо тренувальні дані
mask_std_train = (y_train_raw <= THRESHOLD)
X_train_std = X_train[mask_std_train].copy()
y_train_std_raw = y_train_raw[mask_std_train].copy()

# Розбиваємо дискретні значення на бакети (класи)
# Межі обрані інтуїтивно на основі твого графіка (між основними піками)
bins = [0, 350, 550, 750, 900, 1200]
y_train_std_binned = pd.cut(y_train_std_raw, bins=bins, labels=False)

# Рахуємо медіану в кожному бакеті (це будуть наші очікувані значення для класів)
bucket_medians = y_train_std_raw.groupby(y_train_std_binned).median().to_dict()

# Валідаційна вибірка для ранньої зупинки
mask_std_val = (y_val_raw <= THRESHOLD)
X_val_std = X_val[mask_std_val].copy()
y_val_std_binned = pd.cut(y_val_raw[mask_std_val], bins=bins, labels=False)

std_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(bins)-1,
    n_estimators=1000,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
std_model.fit(
    X_train_std, y_train_std_binned,
    eval_set=[(X_val_std, y_val_std_binned)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)


# --- КРОК 3: Гілка "Довгий хвіст" (Y > 1200) ---
print("\nНавчання регресії для довгого хвоста (Tweedie)...")
mask_tail_train = (y_train_raw > THRESHOLD)
X_train_tail = X_train[mask_tail_train].copy()
y_train_tail = y_train_raw[mask_tail_train].copy()

mask_tail_val = (y_val_raw > THRESHOLD)
X_val_tail = X_val[mask_tail_val].copy()
y_val_tail = y_val_raw[mask_tail_val].copy()

tail_model = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5, # 1.5 - класика для розподілів, схожих на гамма/пуассон
    n_estimators=1000,
    learning_rate=0.03,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
tail_model.fit(
    X_train_tail, y_train_tail,
    eval_set=[(X_val_tail, y_val_tail)],
    eval_metric='mae', # Спостерігаємо за MAE, хоча оптимізуємо Tweedie
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)


# --- КРОК 4: Інференс (Об'єднання прогнозів) ---
print("\nРозрахунок фінальних прогнозів на валідації...")

# 1. Ймовірність того, що клієнт "багатий" (Y > 1200)
p_high = router_model.predict_proba(X_val)[:, 1]

# 2. Очікуваний дохід, якщо клієнт "стандартний"
# Отримуємо ймовірності для кожного бакета (shape: N x num_classes)
std_probs = std_model.predict_proba(X_val) 
expected_std_income = np.zeros(len(X_val))
for i in range(len(bins)-1):
    expected_std_income += std_probs[:, i] * bucket_medians[i]

# 3. Очікуваний дохід, якщо клієнт з "хвоста"
expected_tail_income = tail_model.predict(X_val)

# 4. Фінальна формула математичного сподівання
y_pred_final = (1 - p_high) * expected_std_income + p_high * expected_tail_income

# Оцінка
final_mae = mean_absolute_error(y_val_raw, y_pred_final)
print(f"Фінальний MAE ансамблю: {final_mae:.2f}")