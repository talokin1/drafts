bad_cols = [
    c for c in X_train_proc.columns
    if any(ch in c for ch in ['[', ']', '{', '}', '<', '>', ':', ',', '%', ' '])
]
bad_cols[:10]


import re

def sanitize_feature_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]', '_', c)
        for c in df.columns
    ]
    return df


X_train_proc = sanitize_feature_names(X_train_proc)
X_valid_proc = sanitize_feature_names(X_valid_proc)
