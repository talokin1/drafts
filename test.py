import pandas as pd

def normalize_ids(df, id_cols=("IDENTIFYCODE", "CONTRAGENTID")):
    df = df.copy()

    for col in id_cols:
        if col not in df.columns:
            continue

        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
              .dropna()
              .astype("uint32")
              .astype(str)
              .str.zfill(8)
        )

    return df
