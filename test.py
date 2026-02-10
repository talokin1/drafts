import numpy as np
import pandas as pd
from scipy.stats import skew

# --- Налаштування ---
SKEW_THRESHOLD = 1.0  # Поріг: якщо |skew| > 1, то логарифмуємо
print(f"Skewness threshold set to: {SKEW_THRESHOLD}")

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

# Списки для обробки та їх методи трансформації
# (список колонок, суфікс, функція трансформації)
transform_groups = [
    (real_scale_cols, "_log", signed_log1p),
    (real_diff_cols,  "_slog", signed_log1p),
    (real_raw_cols,   "_log", signed_log1p),
    (real_count_cols, "_log", np.log1p)
]

processed_numeric_feats = []

# --- Логіка Smart Skewness ---
for cols_list, suffix, func in transform_groups:
    for col in cols_list:
        # Рахуємо асиметрію (drop NaN щоб уникнути помилок)
        val_skew = df[col].dropna().skew()
        
        if abs(val_skew) > SKEW_THRESHOLD:
            # Якщо розподіл перекошений -> трансформуємо
            new_col_name = col + suffix
            df[new_col_name] = func(df[col])
            processed_numeric_feats.append(new_col_name)
            # print(f"Transformed {col}: skew={val_skew:.2f} -> {new_col_name}")
        else:
            # Якщо розподіл нормальний -> залишаємо як є
            processed_numeric_feats.append(col)
            # print(f"Kept raw {col}: skew={val_skew:.2f}")

print(f"Total numeric features processed: {len(processed_numeric_feats)}")

# --- Збираємо фінальний список фічей ---
# Додаємо ті, що не проходили через логарифм (ratio, liabs, cat)
features_to_use = (
    processed_numeric_feats +
    real_ratio_cols +
    liabs_cols +
    cat_cols
)

# --- Прибирання ID, якщо він випадково потрапив ---
id_col = 'IDENTIFYCODE'
if id_col in features_to_use:
    features_to_use.remove(id_col)

# --- Формування X та y ---
X = df[features_to_use].copy()
X.index = df[id_col]

y = df["CURR_ACC"]

print(f"FINAL Total Features: {X.shape[1]}")
# Перевірка на дублікати колонок (про всяк випадок)
if len(features_to_use) != len(set(features_to_use)):
    print("Warning: Duplicate features found in list!")