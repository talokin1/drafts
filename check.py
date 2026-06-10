# =========================
# LEAKAGE DIAGNOSTICS
# =========================

print("Rows:", len(df_base))
print("Unique index:", df_base.index.nunique())
print("Index duplicates:", df_base.index.duplicated().sum())

if ID_COL in df_base.columns:
    print("Unique IDENTIFYCODE:", df_base[ID_COL].nunique())
    print("IDENTIFYCODE duplicates:", df_base[ID_COL].duplicated().sum())

# підозрілі фічі по назві
suspicious_keywords = [
    "FX", "IMPORT", "EXPORT", "CURRENCY", "USD", "EUR",
    "PROB_TO_FX", "TARGET", "POTENTIAL"
]

suspicious_features = [
    c for c in final_features
    if any(k.upper() in c.upper() for k in suspicious_keywords)
]

print("\nSuspicious features:")
print(suspicious_features)

# перевірка FX_TYPE, якщо є
if "FX_TYPE" in df_base.columns:
    print("\nFX_TYPE vs target:")
    print(
        pd.crosstab(
            df_base["FX_TYPE"].fillna("NA"),
            (df_base[TARGET_NAME] > 0).astype(int),
            normalize="index"
        )
    )

    print("\nFX_TYPE counts:")
    print(df_base["FX_TYPE"].value_counts(dropna=False))