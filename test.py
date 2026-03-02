# 1. Створюємо ізольований датафрейм
df_val_report = X_val.copy()

# 2. ЖОРСТКЕ ЗШИВАННЯ: додаємо .values, щоб обійти конфлікт індексів
df_val_report[TARGET_NAME] = y_val_raw.values 

# Перевірка, що гроші дійсно перенеслися (має бути > 0)
print(f"Загальна сума доходу у валідації: {df_val_report[TARGET_NAME].sum():.2f}")

# 3. Виклик функції (переконайся, що файл закритий у Excel перед запуском!)
update_business_excel(
    df=df_val_report, 
    df_importance=df_importance, 
    target_col=TARGET_NAME, 
    top_n=4,
    filename="Accounts_Business_Model_Report_v1.xlsx"
)