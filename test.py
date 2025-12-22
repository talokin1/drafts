import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
ID_COLS = ["CONTRAGENTID", "IDENTIFYCODE"]
TARGET_COL = None  # якщо вже є target – впиши назву

# -----------------------------
# BASIC INFO
# -----------------------------
def basic_info(df):
    print("=== BASIC INFO ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\n")

# -----------------------------
# MISSING & ZERO ANALYSIS
# -----------------------------
def missing_zero_analysis(df):
    print("=== MISSING / ZERO ANALYSIS ===")
    res = []
    for col in df.columns:
        if col in ID_COLS:
            continue
        total = len(df)
        nan_cnt = df[col].isna().sum()
        zero_cnt = (df[col] == 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else np.nan
        res.append({
            "feature": col,
            "nan_pct": nan_cnt / total,
            "zero_pct": zero_cnt / total if not np.isnan(zero_cnt) else None,
            "unique": df[col].nunique(dropna=True)
        })
    res_df = pd.DataFrame(res).sort_values("nan_pct", ascending=False)
    print(res_df)
    print("\n")
    return res_df

# -----------------------------
# DISTRIBUTION STATS
# -----------------------------
def distribution_stats(df):
    print("=== DISTRIBUTION STATS ===")
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(ID_COLS)
    desc = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    print(desc)
    print("\n")
    return desc

# -----------------------------
# CONSTANT / LOW VARIANCE
# -----------------------------
def low_variance_features(df, threshold=0.95):
    print("=== CONSTANT / LOW VARIANCE FEATURES ===")
    res = []
    for col in df.columns:
        if col in ID_COLS:
            continue
        top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        res.append({
            "feature": col,
            "top_value_freq": top_freq
        })
    res_df = pd.DataFrame(res)
    bad = res_df[res_df["top_value_freq"] >= threshold].sort_values("top_value_freq", ascending=False)
    print(bad)
    print("\n")
    return bad

# -----------------------------
# CORRELATION (SPEARMAN)
# -----------------------------
def correlation_matrix(df, threshold=0.9):
    print("=== HIGH CORRELATION (SPEARMAN) ===")
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(ID_COLS)
    corr = df[numeric_cols].corr(method="spearman")
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "corr": val
                })
    res_df = pd.DataFrame(pairs).sort_values("corr", key=np.abs, ascending=False)
    print(res_df)
    print("\n")
    return res_df

# -----------------------------
# TARGET LEAKAGE CHECK
# -----------------------------
def target_leakage(df, target):
    if target is None or target not in df.columns:
        print("=== TARGET LEAKAGE CHECK SKIPPED ===\n")
        return None
    print("=== TARGET CORRELATION CHECK ===")
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(ID_COLS + [target])
    corr = df[numeric_cols].corrwith(df[target], method="spearman").sort_values(ascending=False)
    print(corr)
    print("\n")
    return corr

# -----------------------------
# RUN ALL
# -----------------------------
def full_eda(df):
    basic_info(df)
    miss_zero = missing_zero_analysis(df)
    dist = distribution_stats(df)
    low_var = low_variance_features(df)
    corr_pairs = correlation_matrix(df)
    target_corr = target_leakage(df, TARGET_COL)

    return {
        "missing_zero": miss_zero,
        "distribution": dist,
        "low_variance": low_var,
        "correlation_pairs": corr_pairs,
        "target_corr": target_corr
    }

# -----------------------------
# USAGE
# -----------------------------
eda_results = full_eda(df)
