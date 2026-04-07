def normalize_kved(x):
    if pd.isna(x):
        return np.nan
    
    x = str(x).strip()
    
    if '.' in x:
        left, right = x.split('.')
    else:
        left, right = x, ''
    
    left = left.zfill(2)      # 1 → 01
    right = right.ljust(2, '0')  # 2 → 20
    
    return f"{left}.{right[:2]}"

clients['FIRM_KVED'] = clients['FIRM_KVED'].apply(normalize_kved)