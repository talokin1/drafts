# --- Створення хитрих бакетів (ВИПРАВЛЕНО) ---

# 1. Генеруємо основну сітку від 1к до 100к з кроком 5к
main_grid = np.arange(1000, 100001, 5000)

# 2. Збираємо все разом:
# [-1] -> початок для "нульових"
# main_grid -> [1000, 6000, 11000 ...]
# [1000000, np.inf] -> хвіст для мільйонників
bins = [-1] + list(main_grid) + [1000000, np.inf]

# Перевірка на унікальність та сортування (про всяк випадок)
bins = sorted(list(set(bins)))

# Створення лейблів
labels = []
for i in range(len(bins)-1):
    low = bins[i]
    high = bins[i+1]
    
    if high == np.inf:
        labels.append(f"{int(low/1000)}k+")
    elif low == -1:
        # Перший бакет від -1 до 1000 (фактично 0-1k)
        labels.append(f"0-{int(high/1000)}k")
    else:
        labels.append(f"{int(low/1000)}k-{int(high/1000)}k")

# Тепер pd.cut спрацює коректно
report_df['Income_Range'] = pd.cut(report_df['Predicted'], bins=bins, labels=labels)