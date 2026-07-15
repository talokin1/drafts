work = validation_df[
    [propensity_col, final_col, target_col]
].dropna(
    subset=[
        propensity_col,
        final_col,
        target_col
    ]
)

y_true = work[target_col].astype(bool)

if work.empty:
    raise ValueError(
        f"Для {product} немає рядків одночасно "
        f"з propensity-скором і спостереженим таргетом"
    )

if y_true.sum() == 0:
    raise ValueError(
        f"Для {product} немає позитивних клієнтів"
    )