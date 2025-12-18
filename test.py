def is_acquiring_only(series):
    s = series.fillna("").str.lower()

    result = pd.Series(False, index=s.index)

    mask_text = s.str.contains(
        r"(екв|acq|acquiring)",
        regex=True
    )

    mask_2924 = s.str.contains(
        r"(покрит\w*|вируч\w*).{0,40}2924",
        regex=True
    )

    result |= mask_text
    result |= mask_2924

    return result
