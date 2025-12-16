import pandas as pd
import numpy as np
import re

def normalize_fin_report(x):
    if pd.isna(x):
        return "missing"
    
    x = str(x).strip().lower()
    
    if "фінансова звітність" not in x:
        return "missing"
    
    if "мікро" in x:
        return "micro"
    
    if "малого" in x:
        return "small"
    
    if "великого" in x:
        return "large"
    
    if "відсутня" in x:
        return "absent"
    
    return "missing"


df["fin_report_type"] = df["Форма підприємства"].apply(normalize_fin_report)
