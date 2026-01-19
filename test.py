def clip_outliers(df, lower_q=0.01, upper_q=0.99):
    """
    Обрізає 1% екстремальних значень зверху та знизу, замінюючи їх на граничні значення.
    Це краще, ніж видаляти рядки.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        # Не чіпаємо таргет
        if col == 'CURR_ACC': continue
        
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        
        # Замінюємо все, що вище 99-го перцентиля, на значення 99-го перцентиля
        df[col] = df[col].clip(lower=lower, upper=upper)
        
    return df

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

def smart_log_transform(df, threshold=1.0, show_plots=True):
    """
    Автоматично знаходить числові колонки з високим перекосом (skew > threshold)
    та застосовує до них log1p.
    """
    df_transformed = df.copy()
    
    # Вибираємо тільки числові колонки
    numeric_cols = df_transformed.select_dtypes(include=['number']).columns.tolist()
    
    # Виключаємо таргет та бінарні колонки (0/1) з обробки
    exclude_cols = ['CURR_ACC', 'TARGET', 'y_class'] 
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols and df_transformed[c].nunique() > 2]

    print(f"{'Column':<20} | {'Skew (Original)':<15} | {'Action'}")
    print("-" * 50)

    for col in numeric_cols:
        # Рахуємо асиметрію (ігноруємо NaN для розрахунку)
        skew_val = df_transformed[col].skew()
        
        if skew_val > threshold:
            # 1. Застосовуємо логарифм
            # clip(lower=0) гарантує, що не буде помилки log від від'ємного числа
            df_transformed[f'{col}_LOG'] = np.log1p(df_transformed[col].clip(lower=0))
            new_skew = df_transformed[f'{col}_LOG'].skew()
            
            print(f"{col:<20} | {skew_val:.2f}           -> Log Transform (New Skew: {new_skew:.2f})")
            
            # 2. Візуалізація (опціонально)
            if show_plots:
                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                sns.histplot(df[col], bins=30, ax=axes[0], color='salmon')
                axes[0].set_title(f'Original {col} (Skew: {skew_val:.2f})')
                
                sns.histplot(df_transformed[f'{col}_LOG'], bins=30, ax=axes[1], color='skyblue')
                axes[1].set_title(f'Log Transformed (Skew: {new_skew:.2f})')
                plt.tight_layout()
                plt.show()
        else:
            print(f"{col:<20} | {skew_val:.2f}           -> Keep as is")
            
    return df_transformed

# --- Використання ---
# Припускаємо, що df вже завантажено

# Спочатку обрізаємо екстремуми, потім логарифмуємо
df = clip_outliers(df)
df = smart_log_transform(df)

