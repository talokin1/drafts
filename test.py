import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Відфільтруємо рядки, де немає NaN значень, щоб метрики не впали з помилкою
df_clean = df.dropna(subset=['INCOME_LIABILITIES', 'LIABILITIES_POTENTIAL']).copy()

def get_regression_metrics(y_true, y_pred, name="Overall"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Segment": name,
        "Samples_Count": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape * 100, # Переводимо у відсотки для зручності
        "R2": r2
    }

results = []

# 1. Рахуємо загальні метрики по всьому датасету
results.append(get_regression_metrics(
    df_clean['INCOME_LIABILITIES'], 
    df_clean['LIABILITIES_POTENTIAL'], 
    "УСІ ДАНІ (Overall)"
))

# 2. Рахуємо метрики окремо для кожного сегменту бізнесу (MICRO, SMALL, LARGE)
for firm_type in df_clean['FIRM_TYPE'].unique():
    subset = df_clean[df_clean['FIRM_TYPE'] == firm_type]
    results.append(get_regression_metrics(
        subset['INCOME_LIABILITIES'], 
        subset['LIABILITIES_POTENTIAL'], 
        f"Сегмент: {firm_type}"
    ))

# Виводимо результати у вигляді красивого датафрейму
metrics_df = pd.DataFrame(results)
print(metrics_df.to_string(index=False))