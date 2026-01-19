import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sign_log_transform(x):
    """
    Симетричне логарифмування: зберігає знак числа.
    Якщо x = -100 -> результат буде мінусовим логарифмом.
    """
    return np.sign(x) * np.log1p(np.abs(x))

def smart_transform_v2(df, threshold=1.0):
    df_trans = df.copy()
    numeric_cols = df_trans.select_dtypes(include=['number']).columns
    
    # Виключаємо службові колонки
    exclude = ['CURR_ACC', 'TARGET', 'y_class', 'id']
    cols_to_process = [c for c in numeric_cols if c not in exclude and df_trans[c].nunique() > 2]

    print(f"{'Column':<25} | {'Min Val':<10} | {'Action'}")
    print("-" * 60)

    for col in cols_to_process:
        skew_val = df_trans[col].skew()
        min_val = df_trans[col].min()
        
        # Якщо перекос великий (> 1 або < -1)
        if abs(skew_val) > threshold:
            if min_val < 0:
                # ВАРІАНТ 1: Є мінуси (наприклад, Profit або Difference) -> Симетричний лог
                df_trans[f'{col}_LOG'] = sign_log_transform(df_trans[col])
                action = "Symmetric Log (Keep Negatives)"
            else:
                # ВАРІАНТ 2: Тільки позитивні (наприклад, Revenue) -> Звичайний log1p
                df_trans[f'{col}_LOG'] = np.log1p(df_trans[col])
                action = "Standard Log1p"
            
            print(f"{col:<25} | {min_val:<10.2f} | {action}")
        else:
            print(f"{col:<25} | {min_val:<10.2f} | Skip (Normal dist)")
            
    return df_trans

# Застосуйте це замість попереднього
df_clean = smart_transform_v2(df)