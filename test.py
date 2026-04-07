import numpy as np
import pandas as pd

def normalize_kved(x):
    if pd.isna(x):
        return np.nan
    
    x = str(x).strip()
    x = x.replace(',', '.')  # 🔥 ключовий момент

    if '.' in x:
        left, right = x.split('.')
    else:
        left, right = x, ''
    
    left = left.zfill(2)
    right = right.ljust(2, '0')
    
    return f"{left}.{right[:2]}"