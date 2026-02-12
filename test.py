import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# МЕТОД 1: SHAP (Найбільш точний математично)
# Показує "чистий" внесок індустрії, відділений від розміру компанії
# -------------------------------------------------------

# Ініціалізуємо експлейнер (це може зайняти хвилину)
explainer = shap.TreeExplainer(reg)
# Беремо семпл валідації, якщо вона велика, або всю, якщо мала
shap_values = explainer.shap_values(X_val_final)

def analyze_categorical_impact(feature_name, shap_matrix, X_data, top_n=10):
    if feature_name not in X_data.columns:
        print(f"Feature {feature_name} not found!")
        return
    
    # Отримуємо індекс колонки
    col_idx = X_data.columns.get_loc(feature_name)
    
    # Створюємо DF: Сама категорія -> Її SHAP value (вплив на прогноз)
    impact_df = pd.DataFrame({
        'Category': X_data[feature_name].values,
        'SHAP_Impact': shap_matrix[:, col_idx]
    })
    
    # Групуємо та рахуємо середній вплив
    # Positive SHAP = індустрія драйвить дохід вгору
    # Negative SHAP = індустрія тягне дохід вниз
    summary = impact_df.groupby('Category')['SHAP_Impact'].agg(['mean', 'count', 'std'])
    
    # Відфільтруємо зовсім рідкісні категорії для надійності (наприклад, > 50 записів)
    # summary = summary[summary['count'] > 20] 
    
    summary = summary.sort_values('mean', ascending=False)
    
    print(f"\n--- INSIGHTS FOR: {feature_name} ---")
    print(f"TOP-{top_n} Прибуткових (High Potential):")
    print(summary.head(top_n)[['mean', 'count']])
    
    print(f"\nBOTTOM-{top_n} Низько-потенційних:")
    print(summary.tail(top_n)[['mean', 'count']])
    
    # Візуалізація
    plt.figure(figsize=(10, 6))
    top_bottom = pd.concat([summary.head(top_n), summary.tail(top_n)])
    sns.barplot(x=top_bottom['mean'], y=top_bottom.index, palette="RdBu_r")
    plt.title(f"Impact of {feature_name} on Potential Income (SHAP)")
    plt.xlabel("Average Impact on Log(Income)")
    plt.show()

# Запускаємо для Group Code та Division Name
# (Використовуй зрозумілі назви, якщо вони є в X_val, наприклад DIVISION_NAME)
analyze_categorical_impact('GROUP_CODE', shap_values, X_val_final)

# Якщо в X_val є текстова колонка DIVISION_NAME - краще дивитись на неї
if 'DIVISION_NAME' in X_val_final.columns:
    analyze_categorical_impact('DIVISION_NAME', shap_values, X_val_final)
elif 'DIVISION_CODE' in X_val_final.columns:
    analyze_categorical_impact('DIVISION_CODE', shap_values, X_val_final)

# -------------------------------------------------------
# МЕТОД 2: Швидка Агрегація (Business View)
# Просто дивимось, де модель прогнозує найбільші суми в середньому
# -------------------------------------------------------
X_val_analysis = X_val_final.copy()
X_val_analysis['Predicted_Income'] = y_pred_val # (вже експонента)

print("\n--- BUSINESS VIEW: Середній Прогноз по Індустріях ($) ---")
agg_business = X_val_analysis.groupby('GROUP_CODE')['Predicted_Income'].agg(['median', 'mean', 'count'])
print(agg_business.sort_values('median', ascending=False).head(15))