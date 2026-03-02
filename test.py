import pandas as pd
import numpy as np
import os

def update_business_excel(df, df_importance, target_col, top_n=4, filename="Validation_Report.xlsx"):
    # Беремо тільки перші 4 найважливіші ознаки
    top_features = df_importance['Feature'].head(top_n).tolist()
    
    # Режим 'a' (append) додає нові аркуші, не видаляючи старі
    mode = 'a' if os.path.exists(filename) else 'w'
    
    # if_sheet_exists='replace' дозволяє перезаписати аркуш фічі, якщо ти запускаєш код вдруге
    engine_kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
    
    with pd.ExcelWriter(filename, engine='openpyxl', mode=mode, **engine_kwargs) as writer:
        
        for feature in top_features:
            print(f"Обробка фічі: {feature}...")
            
            # 1. Специфічна логіка для кількості співробітників (відновлення після log1p)
            if feature == 'NB_EMPL':
                grouping_col = np.expm1(df[feature]).round().astype(int)
                
            # 2. Логіка для категоріальних ознак (напр., DIVISION_CODE)
            elif pd.api.types.is_categorical_dtype(df[feature]) or df[feature].nunique() < 50:
                grouping_col = df[feature]
                
            # 3. Логіка для інших числових ознак (без інтервалів, просто округлення)
            else:
                grouping_col = df[feature].round(0)
            
            # Агрегація
            agg_df = df.groupby(grouping_col)[target_col].agg(
                Кількість_Клієнтів='count',
                Медіанний_Дохід='median',
                Середній_Дохід='mean',
                Сумарний_Дохід='sum'
            ).reset_index()
            
            # Перейменування колонки для зрозумілості
            agg_df = agg_df.rename(columns={feature: f'{feature}_Value'})
            
            # Сортування: індустрії логічніше сортувати за грошима, а цифри - за зростанням
            if feature == 'DIVISION_CODE':
                agg_df = agg_df.sort_values(by='Сумарний_Дохід', ascending=False)
            else:
                agg_df = agg_df.sort_values(by=f'{feature}_Value', ascending=True)
            
            # Рахуємо відсотки та округлюємо фінанси
            total_income = agg_df['Сумарний_Дохід'].sum()
            agg_df['Частка_в_загальному_доході_%'] = (agg_df['Сумарний_Дохід'] / total_income * 100).round(2)
            
            cols_to_round = ['Медіанний_Дохід', 'Середній_Дохід', 'Сумарний_Дохід']
            agg_df[cols_to_round] = agg_df[cols_to_round].round(2)
            
            # Збереження на окремий аркуш
            sheet_name = str(feature)[:31]
            agg_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"\n✅ Готово! Дані додано у файл: {filename} (старі аркуші збережено)")

# Зверни увагу: зміни назву файлу на ту, де лежить твій Validation_Report
update_business_excel(df_for_report, df_importance, target_col=TARGET_NAME, top_n=4, filename="твоя_назва_файлу.xlsx")