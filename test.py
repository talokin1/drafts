# 1. Готуємо всі компоненти
# Переконайтеся, що y_class існує (з попередніх кроків)
# y_log ви вже створили

# 2. Робіть спліт ОДНОЧАСНО для трьох масивів
# Це гарантує, що індекси будуть синхронізовані
X_train, X_test, y_train_log, y_test_log, y_train_cls, y_test_cls = train_test_split(
    X, 
    y_log, 
    y_class,
    test_size=0.2, 
    random_state=42,
    stratify=y_class # Бажано залишити стратифікацію, щоб VIP-клієнти рівномірно розподілились
)

# 3. Тепер ваші маски працюватимуть ідеально
mask_vip_train = y_train_cls == 1

# Фільтруємо VIP для регресії
X_train_reg = X_train[mask_vip_train]
y_train_reg_log = y_train_log[mask_vip_train]

print(f"Розмір train для регресії (VIP): {X_train_reg.shape}")