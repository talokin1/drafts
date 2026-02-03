# 1. Визначаємо всі числові колонки і відокремлюємо таргет
all_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if "CURR_ACC" in all_numeric_cols:
    all_numeric_cols.remove("CURR_ACC")

# 2. Визначаємо групи (Raw - це все, що не потрапило в твої списки)
# Припускаємо, що списки scale_cols, diff_cols і т.д. вже існують
categorized_cols = set(scale_cols + diff_cols + ratio_cols + count_cols)
raw_cols = [c for c in all_numeric_cols if c not in categorized_cols]

print(f"Categorized features: {len(categorized_cols)}")
print(f"Raw/Uncategorized features found: {len(raw_cols)}")

# 3. Фільтруємо "мертві" колонки (де > 95% нулів)
# Перевіряємо ВСІ колонки разом (і старі, і raw)
cols_to_check = scale_cols + diff_cols + ratio_cols + count_cols + raw_cols
zero_rate = (df[cols_to_check] == 0).mean()
valid_feats = set(zero_rate[zero_rate < 0.95].index)

# Оновлюємо списки - залишаємо тільки "живі"
real_scale_cols = [c for c in scale_cols if c in valid_feats]
real_diff_cols  = [c for c in diff_cols  if c in valid_feats]
real_ratio_cols = [c for c in ratio_cols if c in valid_feats]
real_count_cols = [c for c in count_cols if c in valid_feats]
real_raw_cols   = [c for c in raw_cols   if c in valid_feats]

# 4. Трансформації (через словник, щоб не було PerformanceWarning)
new_feats_data = {}

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

# Signed Log: для грошей, різниць і сирих даних (там можуть бути мінуси)
for col in real_scale_cols + real_diff_cols + real_raw_cols:
    new_feats_data[col + "_log"] = signed_log1p(df[col])

# Log1p: тільки для лічильників (вони >= 0)
for col in real_count_cols:
    new_feats_data[col + "_log"] = np.log1p(df[col])

# Ratios: копіюємо без змін
for col in real_ratio_cols:
    new_feats_data[col] = df[col]

# 5. Збираємо фінальний X одним махом
X = pd.DataFrame(new_feats_data, index=df.index)

# Додаємо категорії (якщо є)
cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
if cat_cols:
    X = pd.concat([X, df[cat_cols]], axis=1) # axis=1 додає стовпці

# 6. Таргет
y = np.log1p(df["CURR_ACC"].clip(lower=0))

print(f"FINAL Total Features: {X.shape[1]}")