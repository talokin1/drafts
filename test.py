import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# 1. Створюємо єдину сітку бакетів для всіх клієнтів
# Межі до 1100 огинають піки. Межі після 1100 захоплюють хвіст.
# Останнє значення np.inf гарантує, що всі "кити" потраплять у крайній бакет
bins = [0, 350, 550, 750, 925, 1100, 2000, 4000, np.inf]

# Перетворюємо неперервний Y у класи (0, 1, 2... K)
y_train_binned = pd.cut(y_train_raw, bins=bins, labels=False)
y_val_binned = pd.cut(y_val_raw, bins=bins, labels=False)

# 2. Рахуємо медіану для КОЖНОГО бакета на трейні
# Це ключовий крок: класифікатор передбачить клас, а ми замінимо його цією медіаною
bucket_medians = y_train_raw.groupby(y_train_binned).median().to_dict()
print("Медіани бакетів (те, що буде прогнозувати модель):")
print(bucket_medians)

# 3. Навчаємо єдиний класифікатор
print("\nНавчання Multiclass моделі...")
clf_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(bins)-1,
    n_estimators=1500,
    learning_rate=0.03,
    class_weight='balanced', # Даємо шанс малим класам з хвоста
    random_state=42,
    n_jobs=-1
)

clf_model.fit(
    X_train, y_train_binned,
    eval_set=[(X_val, y_val_binned)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)

# 4. Прогнозування та розрахунок очікуваного значення
# predict_proba повертає матрицю [N_samples, N_classes]
probs = clf_model.predict_proba(X_val)

y_pred_expected = np.zeros(len(X_val))

# Формула математичного сподівання E[Y] = Sum(P(Class_i) * Median_i)
for i in range(len(bins)-1):
    y_pred_expected += probs[:, i] * bucket_medians[i]

# Оцінка
final_mae = mean_absolute_error(y_val_raw, y_pred_expected)
print(f"\nФінальний MAE (Pure Multiclass): {final_mae:.2f}")

# Для жорсткого прогнозу (якщо бізнесу треба конкретний тариф, а не розмазане очікування)
# Можна брати не зважену суму, а просто медіану найбільш ймовірного класу:
# y_pred_hard = np.array([bucket_medians[c] for c in np.argmax(probs, axis=1)])
# hard_mae = mean_absolute_error(y_val_raw, y_pred_hard)
# print(f"Фінальний MAE (Hard Argmax): {hard_mae:.2f}")