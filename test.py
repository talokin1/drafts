import pandas as pd
import numpy as np

# Припустимо, df_importance - це датафрейм з попереднього кроку
# df - твій оригінальний датафрейм
# TARGET_NAME - назва колонки з реальними комісіями

def create_business_excel(df, df_importance, target_col, top_n=10, filename="Business_Insights_Report.xlsx"):
    # Беремо назви топ-N фічей
    top_features = df_importance['Feature'].head(top_n).tolist()
    
    # Використовуємо ExcelWriter для створення кількох аркушів
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        for feature in top_features:
            print(f"Обробка фічі: {feature}...")
            
            # Перевіряємо, чи фіча є категоріальною (або має мало унікальних значень)
            is_categorical = pd.api.types.is_categorical_dtype(df[feature]) or \
                             pd.api.types.is_object_dtype(df[feature]) or \
                             df[feature].nunique() < 20
            
            if is_categorical:
                # Агрегація для категорій
                agg_df = df.groupby(feature)[target_col].agg(
                    Кількість_Клієнтів='count',
                    Медіанний_Дохід='median',
                    Середній_Дохід='mean',
                    Сумарний_Дохід='sum'
                ).reset_index()
                
                # Сортуємо за сумарним доходом, щоб лідери були зверху
                agg_df = agg_df.sort_values(by='Сумарний_Дохід', ascending=False)
                
            else:
                # Агрегація для неперервних змінних (розбиття на 10 децилів)
                # duplicates='drop' потрібен, якщо багато однакових значень (напр., нулів)
                bins = pd.qcut(df[feature], q=10, duplicates='drop')
                
                agg_df = df.groupby(bins)[target_col].agg(
                    Кількість_Клієнтів='count',
                    Медіанний_Дохід='median',
                    Середній_Дохід='mean',
                    Сумарний_Дохід='sum'
                ).reset_index()
                
                # Перетворюємо інтервали у зрозумілий текст для Excel
                agg_df[feature] = agg_df[feature].astype(str)
                
            # Рахуємо частку (%) від загальної суми для кращого бізнес-розуміння
            total_income = agg_df['Сумарний_Дохід'].sum()
            agg_df['Частка_в_загальному_доході_%'] = (agg_df['Сумарний_Дохід'] / total_income * 100).round(2)
            
            # Округлюємо фінансові колонки до 2 знаків
            cols_to_round = ['Медіанний_Дохід', 'Середній_Дохід', 'Сумарний_Дохід']
            agg_df[cols_to_round] = agg_df[cols_to_round].round(2)
            
            # Зберігаємо на окремий аркуш. Обрізаємо назву до 31 символу (обмеження Excel)
            sheet_name = str(feature)[:31]
            agg_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"\n✅ Готово! Звіт збережено у файл: {filename}")

# Виклик функції
create_business_excel(df, df_importance, TARGET_NAME, top_n=15)