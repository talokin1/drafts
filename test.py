# --- Створення динамічних бакетів ---

# 1. Генеруємо середину: від 1000 до 100000 з кроком 5000
# np.arange(start, stop, step) - не включає останнє число, тому пишемо 100001
mid_bins = np.arange(1000, 100001, 5000) 

# 2. Формуємо повний список меж
# [-1] -> для 0
# [1000] -> кінець першого бакету
# mid_bins -> ваші кроки по 5к (6000, 11000 ... 96000)
# [100000, 1000000, np.inf] -> хвости
raw_bins = [-1] + list(mid_bins) + [100000, 1000000, np.inf]

# Прибираємо дублікати і сортуємо (наприклад, 100к може дублюватися)
bins = sorted(list(set(raw_bins)))

# 3. Автоматично створюємо підписи (Labels), щоб не писати їх вручну
labels = []
for i in range(len(bins) - 1):
    low = bins[i]
    high = bins[i+1]
    
    if low == -1:
        labels.append("0-1k")
    elif high == np.inf:
        labels.append(f"{int(low/1000)}k+")
    else:
        # Формат: "1k-6k", "6k-11k" і т.д.
        labels.append(f"{int(low/1000)}k-{int(high/1000)}k")

# Застосовуємо (переконайтеся, що колонка називається 'Fcst', як на скріні)
final_report['Income_Range'] = pd.cut(final_report['Fcst'], bins=bins, labels=labels)