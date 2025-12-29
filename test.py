def drop_leakage_by_name(df, target):
    drop_cols = []
    for c in df.columns:
        if c == target:
            continue
        if any(k in c.upper() for k in LEAKAGE_KEYWORDS):
            drop_cols.append(c)

    df = df.drop(columns=drop_cols, errors="ignore")
    return df, drop_cols
