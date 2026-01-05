raw_numbers = (
    df
    .apply(pd.to_numeric, errors='coerce')  # усе → float або NaN
    .stack()                                # в один Series
    .dropna()                               # прибираємо NaN
    .tolist()
)
