def drop_id_cols(df):
    return df.drop(columns=ID_COLS, errors="ignore")

def select_feature_versions(df):
    groups = {}
    for c in df.columns:
        if "_" not in c:
            continue
        base = c.rsplit("_", 1)[0]
        groups.setdefault(base, []).append(c)

    drop_cols = []

    for base, cols in groups.items():
        if len(cols) == 1:
            continue

        keep = None
        for suf in SUFFIX_PRIORITY:
            for c in cols:
                if c.endswith(suf):
                    keep = c
                    break
            if keep:
                break

        if keep is None:
            keep = cols[0]

        drop_cols.extend([c for c in cols if c != keep])

    df = df.drop(columns=drop_cols, errors="ignore")
    return df, drop_cols


def drop_full_zero_columns(df):
    zero_cols = [
        c for c in df.columns
        if df[c].dtype.kind in "biufc" and (df[c] == 0).all()
    ]
    df = df.drop(columns=zero_cols, errors="ignore")
    return df, zero_cols


def structural_preprocess(df):
    df = df.copy()

    df = drop_id_cols(df)
    df, dropped_leakage = drop_leakage_by_name(df, TARGET_COL)
    df, dropped_versions = select_feature_versions(df)
    df, dropped_zero = drop_full_zero_columns(df)

    meta = {
        "dropped_leakage": dropped_leakage,
        "dropped_versions": dropped_versions,
        "dropped_zero": dropped_zero
    }

    return df, meta

df_clean, structural_meta = structural_preprocess(main_dataset)

print(df_clean.shape)


from sklearn.model_selection import train_test_split

X = df_clean.drop(columns=[TARGET_COL])
y = df_clean[TARGET_COL]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42
)

X_train.reset_index(drop=True, inplace=True)
X_valid.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
