# 1. Створюємо ізольований датафрейм виключно з валідаційної вибірки
df_val_report = X_val.copy()

# 2. Повертаємо колонку з фактичним доходом на своє місце
# y_val_raw містить оригінальні грошові значення (не бакети)
df_val_report[TARGET_NAME] = y_val_raw

# Перевірка надійності, щоб упевнитись, що змінна Accounts точно на місці
print(f"Формат валідаційних даних: {df_val_report.shape}")
print(f"Чи є колонка '{TARGET_NAME}' у df_val_report? {'Так' if TARGET_NAME in df_val_report.columns else 'Ні'}")

# 3. Викликаємо функцію, передаючи їй валідаційний датафрейм
update_business_excel(
    df=df_val_report, 
    df_importance=df_importance, 
    target_col=TARGET_NAME, 
    top_n=4,
    filename="Accounts_Business_Model_Report_v1.xlsx" # Вкажи тут назву свого файлу
)