# 1. Беремо ВСІ числові колонки з датафрейму
all_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Викидаємо таргет, якщо він там є
if "CURR_ACC" in all_numeric_cols:
    all_numeric_cols.remove("CURR_ACC")

# 2. Збираємо ті, що ми вже класифікували
categorized_cols = set(scale_cols + diff_cols + ratio_cols + count_cols)

# 3. Знаходимо різницю (це і є твої втрачені 300+ фічей)
raw_cols = [c for c in all_numeric_cols if c not in categorized_cols]

print(f"Categorized features: {len(categorized_cols)}")
print(f"Raw/Uncategorized features found: {len(raw_cols)}") # Тут має бути ~300
print(f"Example raw cols: {raw_cols[:5]}")

# -------------------------------------------------------
# ОНОВЛЕНИЙ ПАЙПЛАЙН ЗБОРКИ
# -------------------------------------------------------

# Додаємо raw_cols до перевірки на нулі
num_cols_all = scale_cols + diff_cols + ratio_cols + count_cols + raw_cols
zero_rate = (df[num_cols_all] == 0).mean()
valid_feats = set(zero_rate[zero_rate < 0.99].index) # Жорсткий фільтр

# Фільтруємо списки
real_scale_cols = [c for c in scale_cols if c in valid_feats]
real_diff_cols  = [c for c in diff_cols  if c in valid_feats]
real_ratio_cols = [c for c in ratio_cols if c in valid_feats]
real_count_cols = [c for c in count_cols if c in valid_feats]
real_raw_cols   = [c for c in raw_cols   if c in valid_feats] # Фільтруємо raw

# Трансформації
# ... (твої попередні цикли для scale/diff/count) ...

# Для RAW колонок використовуємо Signed Log (найбезпечніший варіант)
for col in real_raw_cols:
    df[col + "_log"] = signed_log1p(df[col])

# Фінальний список
features_to_use = (
    [c + "_log"  for c in real_scale_cols] +
    [c + "_slog" for c in real_diff_cols] +
    [c + "_log"  for c in real_count_cols] +
    [c + "_log"  for c in real_raw_cols] + # Додаємо raw
    real_ratio_cols +
    cat_cols
)

X = df[features_to_use].copy()
print(f"FINAL Total Features: {X.shape[1]}")