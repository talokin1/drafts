def target_distribution(df, target):
    print("=== TARGET DISTRIBUTION ===")
    
    s = df[target]
    print(s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
    
    zero_pct = (s == 0).mean()
    neg_pct = (s < 0).mean()
    
    print(f"\nZero %: {zero_pct:.3f}")
    print(f"Negative %: {neg_pct:.3f}\n")
    
    return {
        "describe": s.describe(),
        "zero_pct": zero_pct,
        "negative_pct": neg_pct
    }

def feature_suffix_groups(df):
    print("=== FEATURE GROUPS (CUR / PREV / NORM) ===")
    
    groups = {}
    for c in df.columns:
        if "_" not in c:
            continue
        base = c.rsplit("_", 1)[0]
        groups.setdefault(base, []).append(c)
    
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    
    for k, v in list(multi.items())[:20]:
        print(f"{k}: {v}")
    
    print(f"\nTotal multi-version groups: {len(multi)}\n")
    return multi

def sparsity_analysis(df, threshold=0.95):
    print("=== SPARSITY ANALYSIS ===")
    
    res = []
    for col in df.select_dtypes(include=np.number).columns:
        zero_ratio = (df[col] == 0).mean()
        res.append({
            "feature": col,
            "zero_ratio": zero_ratio
        })
    
    res_df = pd.DataFrame(res)
    sparse = res_df[res_df["zero_ratio"] >= threshold] \
        .sort_values("zero_ratio", ascending=False)
    
    print(sparse.head(20))
    print("\n")
    return sparse


def leakage_name_check(df, target):
    print("=== LEAKAGE NAME CHECK ===")
    
    suspicious = [
        c for c in df.columns
        if any(k in c.lower() for k in ["income", "revenue", "profit"])
        and c != target
    ]
    
    print(suspicious)
    print("\n")
    return suspicious


def full_eda(df):
    basic_info(df)
    
    miss_zero = missing_zero_analysis(df)
    dist = distribution_stats(df)
    low_var = low_variance_features(df)
    corr_pairs = correlation_matrix(df)
    
    target_corr = target_leakage(df, TARGET_COL)
    target_dist = target_distribution(df, TARGET_COL)
    
    suffix_groups = feature_suffix_groups(df)
    sparse = sparsity_analysis(df)
    mono = monotonicity_check(df, TARGET_COL)
    leakage_names = leakage_name_check(df, TARGET_COL)
    
    return {
        "missing_zero": miss_zero,
        "distribution": dist,
        "low_variance": low_var,
        "correlation_pairs": corr_pairs,
        "target_corr": target_corr,
        "target_distribution": target_dist,
        "suffix_groups": suffix_groups,
        "sparsity": sparse,
        "monotonicity": mono,
        "leakage_names": leakage_names
    }
