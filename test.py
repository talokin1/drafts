def normalize_kved(x):
    if not isinstance(x, str):
        return None
    x = x.strip()

    # 91.31.0 → 91.31
    if x.count(".") >= 2:
        x = ".".join(x.split(".")[:2])

    return x
df["KVED_NORM"] = df["KVED"].apply(normalize_kved)
