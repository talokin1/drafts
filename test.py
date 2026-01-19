baseline = df.loc[X_val.index, TARGET_COL].median()
mae_baseline = mean_absolute_error(
    df.loc[X_val.index, TARGET_COL],
    np.full(len(X_val), baseline)
)
mae_baseline
