import numpy as np

def fit_numeric_stats(X_train, clip_low=0.01, clip_high=0.99):
    stats = {}

    num_cols = X_train.select_dtypes(include=np.number).columns

    for c in num_cols:
        s = X_train[c]
        stats[c] = {
            "median": s.median(),
            "clip_low": s.quantile(clip_low),
            "clip_high": s.quantile(clip_high)
        }

    return stats


def apply_numeric_preprocess(X, stats):
    X = X.copy()

    for c, p in stats.items():
        if c not in X.columns:
            continue

        X[c] = X[c].fillna(p["median"])
        X[c] = X[c].clip(p["clip_low"], p["clip_high"])

        if c.endswith("_DIF"):
            X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))

    return X


def apply_numeric_preprocess(X, stats):
    X = X.copy()

    for c, p in stats.items():
        if c not in X.columns:
            continue

        X[c] = X[c].fillna(p["median"])
        X[c] = X[c].clip(p["clip_low"], p["clip_high"])

        if c.endswith("_DIF"):
            X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))

    return X

def fit_categories(X_train):
    cat_info = {}

    cat_cols = X_train.select_dtypes(include="object").columns

    for c in cat_cols:
        cat_info[c] = set(
            X_train[c].fillna("unknown").astype(str).unique()
        )

    return cat_info



def apply_categories(X, cat_info):
    X = X.copy()

    for c, allowed in cat_info.items():
        if c not in X.columns:
            continue

        X[c] = X[c].fillna("unknown").astype(str)
        X[c] = X[c].where(X[c].isin(allowed), "unknown")
        X[c] = X[c].astype("category")

    return X


# FIT тільки на train
num_stats = fit_numeric_stats(X_train)
cat_info = fit_categories(X_train)

# APPLY
X_train_proc = apply_numeric_preprocess(X_train, num_stats)
X_train_proc = apply_categories(X_train_proc, cat_info)

X_valid_proc = apply_numeric_preprocess(X_valid, num_stats)
X_valid_proc = apply_categories(X_valid_proc, cat_info)
