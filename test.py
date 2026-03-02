import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# 1. Ізолюємо фічі та таргет (без старих колонок маршрутизатора)
# Припускаємо, що df - це актуальний чистий датафрейм
y_raw = df[TARGET_NAME].copy()
X = df.drop(columns=[TARGET_NAME]) 

# Видаляємо старі колонки, якщо вони випадково лишилися в X
if 'is_high_income' in X.columns:
    X = X.drop(columns=['is_high_income'])

# 2. Створюємо єдину сітку бакетів
# Починаємо з -1, щоб гарантовано захопити нулі, якщо вони десь проскочили
bins = [-1, 350, 550, 750, 925, 1100, 2000, 4000, np.inf]

# Перетворюємо неперервний Y у класи (0, 1, 2... K)
y_binned = pd.cut(y_raw, bins=bins, labels=False)

# 3. Розбиття на train/val зі стратифікацією по БАКЕТАХ
X_train, X_val, y_train_raw, y_val_raw, y_train_binned, y_val_binned = train_test_split(
    X, y_raw, y_binned, 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
    stratify=y_binned  # Стратифікація за класами гарантує стабільність метрик
)

# Перетворення категоріальних фічей для LightGBM
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

# 4. Рахуємо медіани бакетів тільки на TRAIN вибірці
bucket_medians = y_train_raw.groupby(y_train_binned).median().to_dict()
print("Медіани бакетів (очікувані значення класів):")
print(bucket_medians)

# 5. Навчання Multiclass моделі
print("\nНавчання Pure Multiclass моделі...")
clf_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(bins)-1,
    n_estimators=1500,
    learning_rate=0.03,
    class_weight='balanced', # Змушуємо модель поважати класи з "хвоста"
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf_model.fit(
    X_train, y_train_binned,
    eval_set=[(X_val, y_val_binned)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)

# 6. Розрахунок очікуваного доходу (Математичне сподівання)
probs = clf_model.predict_proba(X_val)

y_pred_expected = np.zeros(len(X_val))

for i in range(len(bins)-1):
    y_pred_expected += probs[:, i] * bucket_medians[i]

# Фінальна метрика
final_mae = mean_absolute_error(y_val_raw, y_pred_expected)
print(f"\nФінальний MAE (Pure Multiclass E[Y]): {final_mae:.2f}")

# Альтернатива: Жорсткий прогноз найімовірнішого тарифу
y_pred_hard = np.array([bucket_medians[c] for c in np.argmax(probs, axis=1)])
hard_mae = mean_absolute_error(y_val_raw, y_pred_hard)
print(f"Фінальний MAE (Hard Argmax): {hard_mae:.2f}")