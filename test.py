import pandas as pd
import numpy as np

def generate_specific_business_report(df, features_list, target_col, filename="Regression_Business_Report.xlsx"):
    # Створюємо новий Excel файл з перезаписом
    with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
        
        for feature in features_list:
            print(f"Обробка фічі: {feature}...")
            
            # Перевірка на випадок, якщо назва фічі трохи відрізняється (напр. REVENUE_CURR vs REVENUE_CUR)
            if feature not in df.columns:
                print(f"⚠️ Ознаки '{feature}' немає в датасеті. Пропускаємо.")
                continue
                
            # 1. Логіка для категоріальних ознак (Сектор економіки)
            if feature == 'SECTION_CODE' or df[feature].nunique() < 50:
                grouping_col = df[feature]
                
            # 2. Логіка для фінансових/неперервних ознак (розбиття на децилі)
            else:
                # qcut ділить вибірку на 10 рівних за кількістю клієнтів частин
                grouping_col = pd.qcut(df[feature], q=10, duplicates='drop')
                
            # Агрегація
            agg_df = df.groupby(grouping_col, observed=False)[target_col].agg(
                Кількість_Клієнтів='count',
                Медіанний_Дохід='median',
                Середній_Дохід='mean',
                Сумарний_Дохід='sum'
            ).reset_index()
            
            # Перейменування колонки для зрозумілості в Excel
            agg_df = agg_df.rename(columns={feature: f'{feature}_Value'})
            
            # Сортування: категоріальні сортуємо за грошима (лідери зверху)
            if feature == 'SECTION_CODE':
                agg_df = agg_df.sort_values(by='Сумарний_Дохід', ascending=False)
            else:
                # Для квантилів перетворюємо інтервали в текст, щоб Excel їх не зламав
                agg_df[f'{feature}_Value'] = agg_df[f'{feature}_Value'].astype(str)
            
            # Рахуємо фінансові частки
            total_income = agg_df['Сумарний_Дохід'].sum()
            agg_df['Частка_в_загальному_доході_%'] = (agg_df['Сумарний_Дохід'] / total_income * 100).round(2)
            
            cols_to_round = ['Медіанний_Дохід', 'Середній_Дохід', 'Сумарний_Дохід']
            agg_df[cols_to_round] = agg_df[cols_to_round].round(2)
            
            # Зберігаємо на окремий аркуш
            sheet_name = str(feature)[:31]
            agg_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"\n✅ Готово! Звіт збережено у файл: {filename}")

# --- ВИКЛИК ФУНКЦІЇ ---

# 1. Створюємо тимчасовий датафрейм (заміни X_val на свою змінну з ознаками)
df_report_reg = X_val.copy()

# 2. ЖОРСТКЕ ЗШИВАННЯ (.values) цільової змінної. 
# Важливо: використовуй зміну зі справжніми сумами (НЕ логарифмованими, якщо робив log1p)
df_report_reg[TARGET_NAME] = y_val_raw.values 

# 3. Список фічей (на графіку у тебе REVENUE_CUR, без подвійного R на кінці)
features_to_analyze = ['REVENUE_CUR', 'SECTION_CODE', 'B1600', 'PRIMARY_ASSETS']

# 4. Запускаємо генерацію
generate_specific_business_report(
    df=df_report_reg,
    features_list=features_to_analyze,
    target_col=TARGET_NAME,
    filename="Regression_Specific_Features_Report.xlsx"
)