import pandas as pd
import numpy as np
import os

def generate_specific_business_report(df, features_list, target_col, filename="Assets_Business_Model_Report_v1.xlsx"):
    # Режим 'a' гарантує, що старі аркуші (Validation_Report) залишаться недоторканими
    mode = 'a' if os.path.exists(filename) else 'w'
    engine_kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
    
    with pd.ExcelWriter(filename, engine='openpyxl', mode=mode, **engine_kwargs) as writer:
        
        for feature in features_list:
            print(f"Обробка фічі: {feature}...")
            
            if feature not in df.columns:
                continue
                
            is_categorical = (
                isinstance(df[feature].dtype, pd.CategoricalDtype) or 
                pd.api.types.is_object_dtype(df[feature]) or 
                feature in ['SECTION_CODE', 'DIVISION_CODE'] or
                df[feature].nunique() < 50
            )
            
            if is_categorical:
                # Класичне групування для категорій
                grouping_col = df[feature]
                agg_df = df.groupby(grouping_col, observed=False)[target_col].agg(
                    Кількість_Клієнтів='count',
                    Медіанний_Дохід='median',
                    Середній_Дохід='mean',
                    Сумарний_Дохід='sum'
                ).reset_index()
                
                agg_df = agg_df.rename(columns={feature: f'{feature}_Value'})
                agg_df = agg_df.sort_values(by='Сумарний_Дохід', ascending=False)
                
            else:
                # 1. Повертаємо логарифми у реальні грошові суми
                real_values = np.expm1(df[feature])
                
                # 2. Розбиваємо вибірку на 10 сегментів
                buckets = pd.qcut(real_values, q=10, duplicates='drop')
                
                # 3. Агрегуємо фінансові метрики для кожного сегмента
                agg_df = df.groupby(buckets, observed=False).agg(
                    Кількість_Клієнтів=(target_col, 'count'),
                    Медіанний_Дохід=(target_col, 'median'),
                    Середній_Дохід=(target_col, 'mean'),
                    Сумарний_Дохід=(target_col, 'sum')
                )
                
                # 4. КЛЮЧОВИЙ МОМЕНТ: замінюємо інтервал на єдину конкретну цифру 
                # (медіану активів/виручки в цьому сегменті)
                bucket_medians = real_values.groupby(buckets, observed=False).median()
                agg_df.index = bucket_medians.round(0).astype(int).values
                agg_df.index.name = f'{feature}_Value'
                agg_df = agg_df.reset_index()
                
                agg_df = agg_df.sort_values(by=f'{feature}_Value', ascending=True)
                
            # Загальні фінансові розрахунки
            total_income = agg_df['Сумарний_Дохід'].sum()
            agg_df['Частка_в_загальному_доході_%'] = (agg_df['Сумарний_Дохід'] / total_income * 100).round(2)
            
            cols_to_round = ['Медіанний_Дохід', 'Середній_Дохід', 'Сумарний_Дохід']
            agg_df[cols_to_round] = agg_df[cols_to_round].round(2)
            
            sheet_name = str(feature)[:31]
            agg_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"\n✅ Готово! Звіт доповнено (старі аркуші збережено): {filename}")

# --- ВИКЛИК ---
# Переконайся, що файл закритий у Excel, інакше виникне Permission Error
features_to_analyze = ['REVENUE_CUR', 'DIVISION_CODE', 'B1600', 'PRIMARY_ASSETS']

generate_specific_business_report(
    df=df_report_reg, 
    features_list=features_to_analyze, 
    target_col=TARGET_NAME, 
    filename="Твій_Файл_З_Validation_Report.xlsx" 
)