import numpy as np
import pandas as pd

# ---------------------------------------------------------
# КРОК 1: Визначення валідних фічей (Cleaning)
# ---------------------------------------------------------
# Рахуємо відсоток нулів тільки для числових колонок
num_cols_all = scale_cols + diff_cols + ratio_cols + count_cols
zero_rate = (df[num_cols_all] == 0).mean()

# Залишаємо тільки ті, де нулів менше 99% (або 95%, як ти вирішив)
valid_feats = set(zero_rate[zero_rate < 0.99].index)

# Оновлюємо списки колонок - тепер у них тільки "живі" дані
real_scale_cols = [c for c in scale_cols if c in valid_feats]
real_diff_cols  = [c for c in diff_cols  if c in valid_feats]
real_ratio_cols = [c for c in ratio_cols if c in valid_feats]
real_count_cols = [c for c in count_cols if c in valid_feats]

# ---------------------------------------------------------
# КРОК 2: Математичні трансформації
# ---------------------------------------------------------
# Функція Signed Log (рятує мінуси)
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

# 1. SCALE COLS (Тут можуть бути мінуси в прибутку!)
for col in real_scale_cols:
    # Перевірка: якщо колонка строго додатна (як виручка), можна log1p
    # Але signed_log1p безпечніший, якщо ти не впевнений у кожній колонці
    df[col + "_log"] = signed_log1p(df[col])

# 2. DIFF COLS (Тут точно є мінуси)
for col in real_diff_cols:
    df[col + "_slog"] = signed_log1p(df[col])

# 3. RATIO COLS
# Зазвичай їх не треба логарифмувати, якщо це відсотки (-1..1) або прапорці (0/1).
# Просто копіюємо їх у фінальний набір (або залишаємо як є).
# Якщо там є гігантські значення (наприклад, leverage 1000%), тоді логарифмуй.
# Припустимо, що вони нормальні:
final_ratio_cols = real_ratio_cols 

# 4. COUNT COLS (Співробітники - завжди +)
for col in real_count_cols:
    df[col + "_log"] = np.log1p(df[col])

# ---------------------------------------------------------
# КРОК 3: Фінальна збірка (Assembly)
# ---------------------------------------------------------
# Збираємо імена нових колонок
features_to_use = (
    [c + "_log"  for c in real_scale_cols] +
    [c + "_slog" for c in real_diff_cols] +
    [c + "_log"  for c in real_count_cols] +
    final_ratio_cols +  # Оригінальні
    cat_cols            # Категорії (LGBM сам розбереться або треба кодувати)
)

# Створюємо чистий X та y
# (Припускаємо, що таргет вже оброблений окремо)
X = df[features_to_use].copy()
y = df["CURR_ACC"] 

print(f"Original numeric features: {len(num_cols_all)}")
print(f"Features after zero-filter: {len(valid_feats)}")
print(f"Final feature count (including cats): {X.shape[1]}")