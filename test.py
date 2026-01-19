import re
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# -----------------------------
# 1) Helpers: auto-detect columns
# -----------------------------
RATIO_PAT = re.compile(r"(RATIO|MARGIN|SHARE|PCT|PERCENT|%|_NORM$|_NORM_|_RATE$|_RATE_|_IDX$|_INDEX$)", re.IGNORECASE)
DIFF_PAT  = re.compile(r"(_DIF$|_DIFF$|DIF_|DIFF_|DELTA|CHANGE|CHG)", re.IGNORECASE)
PREV_PAT  = re.compile(r"(_PREV$|_PRV$|PREV_|PRV_|PREVIOUS)", re.IGNORECASE)
IDLIKE_PAT = re.compile(r"(ID$|_ID$|CODE$|_CODE$|KVED|OPCD|EDRPOU|TAX|IBAN|ACCOUNT|ACC|MASK)", re.IGNORECASE)
COUNTLIKE_PAT = re.compile(r"(NB_|NUM_|N_|COUNT|CNT|QTY|EMP|EMPL|STAFF)", re.IGNORECASE)


def is_binary_series(s: pd.Series) -> bool:
    # allow NaN-free binary
    vals = pd.unique(s)
    return len(vals) <= 2 and set(vals).issubset({0, 1})


def heavy_tail_flag(s: pd.Series) -> bool:
    # robust heavy-tail test: huge skew between 99% and 50% OR mean >> median
    q50 = s.quantile(0.50)
    q99 = s.quantile(0.99)
    mean = s.mean()
    # avoid division by zero
    denom = abs(q50) + 1e-9
    return (abs(q99) / denom > 50) or (abs(mean) / (abs(q50) + 1e-9) > 20)


def detect_feature_groups(df: pd.DataFrame, target_col: str):
    cols = [c for c in df.columns if c != target_col]

    # categorical: existing category/object + id-like columns with low-ish cardinality
    cat_cols = []
    for c in cols:
        if pd.api.types.is_object_dtype(df[c]) or str(df[c].dtype) == "category":
            cat_cols.append(c)
            continue

        # treat code-like ints as categorical if not too many uniques
        if IDLIKE_PAT.search(c) and pd.api.types.is_numeric_dtype(df[c]):
            nun = df[c].nunique(dropna=False)
            if nun <= 5000:  # safe threshold for 31k rows
                cat_cols.append(c)

    # numeric candidates
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c not in cat_cols]

    # name-based groups
    diff_cols = [c for c in num_cols if DIFF_PAT.search(c)]
    prev_cols = [c for c in num_cols if PREV_PAT.search(c)]

    ratio_cols = [c for c in num_cols if RATIO_PAT.search(c)]  # keep as-is
    bin_cols = [c for c in num_cols if is_binary_series(df[c])]  # keep as-is

    # count-like: integer dtypes or name pattern, excluding binary and ratio/diff/prev
    count_cols = []
    for c in num_cols:
        if c in bin_cols or c in ratio_cols or c in diff_cols or c in prev_cols:
            continue
        if pd.api.types.is_integer_dtype(df[c]) or COUNTLIKE_PAT.search(c):
            # but avoid treating huge-cardinality continuous ints as counts
            nun = df[c].nunique()
            if nun <= 20000:
                count_cols.append(c)

    # scale-like: remaining numeric columns (excluding diff/prev/ratio/binary/count)
    rest = [c for c in num_cols if c not in set(diff_cols + prev_cols + ratio_cols + bin_cols + count_cols)]
    # choose scale columns as heavy-tail among rest
    scale_cols = [c for c in rest if heavy_tail_flag(df[c])]

    # Anything left in rest but not scale -> "other_numeric" keep as-is (trees handle it)
    other_numeric = [c for c in rest if c not in scale_cols]

    # final cleanup: remove target if mistakenly added
    for lst in (cat_cols, num_cols, diff_cols, prev_cols, ratio_cols, bin_cols, count_cols, scale_cols, other_numeric):
        if target_col in lst:
            lst.remove(target_col)

    # ensure unique
    cat_cols = list(dict.fromkeys(cat_cols))

    return {
        "cat_cols": cat_cols,
        "diff_cols": diff_cols,
        "prev_cols": prev_cols,
        "ratio_cols": ratio_cols,
        "bin_cols": bin_cols,
        "count_cols": count_cols,
        "scale_cols": scale_cols,
        "other_numeric": other_numeric,
        "num_cols": num_cols,
    }


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


# -----------------------------
# 2) Pipeline: prep -> train -> MAE
# -----------------------------
def train_lgbm_pipeline(df: pd.DataFrame, target_col: str, min_target: float = 100.0, test_size: float = 0.2):
    df = df.copy()

    # trim as you decided
    df = df[df[target_col] > min_target].copy()

    # target in log space (MAE evaluated after inverse)
    y = np.log1p(df[target_col])

    groups = detect_feature_groups(df, target_col=target_col)

    # set categorical dtype
    for c in groups["cat_cols"]:
        df[c] = df[c].astype("category")

    # create transformed features
    created = []

    # scale -> log1p (clip to >=0)
    for c in groups["scale_cols"]:
        newc = f"{c}__log"
        df[newc] = np.log1p(df[c].clip(lower=0))
        created.append(newc)

    # diffs -> signed log
    for c in groups["diff_cols"]:
        newc = f"{c}__slog"
        df[newc] = signed_log1p(df[c])
        created.append(newc)

    # counts -> log1p (optional but usually good)
    for c in groups["count_cols"]:
        newc = f"{c}__log"
        df[newc] = np.log1p(df[c].clip(lower=0))
        created.append(newc)

    # drop prev cols (soft dedup)
    df.drop(columns=groups["prev_cols"], inplace=True, errors="ignore")

    # optionally drop raw versions of transformed cols to reduce collinearity/noise
    df.drop(columns=groups["scale_cols"] + groups["diff_cols"] + groups["count_cols"], inplace=True, errors="ignore")

    # assemble final feature list
    features = (
        created
        + groups["ratio_cols"]
        + groups["bin_cols"]
        + groups["other_numeric"]
        + groups["cat_cols"]
    )

    # remove any missing features due to drops
    features = [c for c in features if c in df.columns]

    X = df[features]
    y = y.loc[X.index]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="mae",
        categorical_feature=groups["cat_cols"],
        verbose=100
    )

    # MAE in original space
    y_pred = np.expm1(model.predict(X_valid))
    y_true = np.expm1(y_valid)
    mae = mean_absolute_error(y_true, y_pred)

    return model, groups, features, mae


# -----------------------------
# 3) Usage
# -----------------------------
target_col = "CURR_ACC"
model, groups, features, mae = train_lgbm_pipeline(df, target_col=target_col, min_target=100.0)

print("MAE:", mae)
print("\nDetected groups:")
for k, v in groups.items():
    if isinstance(v, list):
        print(f"{k}: {len(v)}")
