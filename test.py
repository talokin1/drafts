# Ensure ID exists
if ID_COL not in df.columns:
    df = df.reset_index().rename(columns={"index": ID_COL})

# Merge industry + OPF meta from X_val
if ID_COL not in X_val.columns and X_val.index.name == ID_COL:
    df = df.merge(
        X_val[[INDUSTRY_COL, OPF_COL]],
        left_on=ID_COL,
        right_index=True,
        how="left"
    )
else:
    df = df.merge(
        X_val[[ID_COL, INDUSTRY_COL, OPF_COL]],
        on=ID_COL,
        how="left"
    )
