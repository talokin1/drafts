ID_COLS = ["CONTRAGENTID", "IDENTIFYCODE"]

LEAKAGE_KEYWORDS = [
    "REVENUE",
    "GROSS_PROFIT",
    "OPER_PROFIT",
    "NET_PROFIT",
    "INCOME"
]

SUFFIX_PRIORITY = ["_NORM", "_CUR", "_DIF", "_PREV"]


def drop_leakage(df, target):
    cols = []
    for c in df.columns:
        if c == target:
            continue
        if any(k in c.upper() for k in LEAKAGE_KEYWORDS):
            cols.append(c)
    return df.drop(columns=cols), cols


def select_feature_versions(df):
    groups = {}
    for c in df.columns:
        if "_" not in c:
            continue
        base = c.rsplit("_", 1)[0]
        groups.setdefault(base, []).append(c)

    drop = []
    for base, cols in groups.items():
        if len(cols) == 1:
            continue

        for suf in SUFFIX_PRIORITY:
            selected = [c for c in cols if c.endswith(suf)]
            if selected:
                keep = selected[0]
                break
        else:
            keep = cols[0]

        drop.extend([c for c in cols if c != keep])

    return df.drop(columns=drop), drop


def drop_full_zero(df):
    drop = [
        c for c in df.columns
        if df[c].dtype.kind in "biufc" and (df[c] == 0).all()
    ]
    return df.drop(columns=drop), drop

def numeric_cleaning(df):
    num_cols = df.select_dtypes(include=np.number).columns

    # NaN → median
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # clip
    for c in num_cols:
        lo, hi = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lo, hi)

    # signed log for DIF
    for c in num_cols:
        if c.endswith("_DIF"):
            df[c] = np.sign(df[c]) * np.log1p(np.abs(df[c]))

    return df


def prepare_categories(df):
    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        df[c] = df[c].fillna("unknown").astype("category")
    return df, list(cat_cols)



df = main_dataset.copy()

# 1. Drop ID
df = df.drop(columns=ID_COLS, errors="ignore")

# 2. Drop leakage (by name)
df, _ = drop_leakage(df, TARGET_COL)

# 3. CUR / PREV / DIF selection
df, _ = select_feature_versions(df)

# 4. Drop full-zero columns
df, _ = drop_full_zero(df)



X_train_proc = preprocess_numeric_fit(X_train)
X_test_proc = preprocess_numeric_transform(X_test)
