import pandas as pd
import re
import numpy as np

df = df.copy()

LEVELS = {"низький", "середній", "високий"}
DATE_RE = re.compile(r"\d{2}\.\d{2}\.\d{4}")

def is_score(x):
    try:
        x = float(x)
        return 50 <= x <= 500
    except:
        return False

def extract_msb_from_row(row):
    score = None
    level = None
    date = None

    for val in row.values:
        if pd.isna(val):
            continue

        s = str(val).strip().lower()

        if score is None and is_score(s):
            score = float(s)
            continue

        if level is None and s in LEVELS:
            level = s
            continue

        if date is None and DATE_RE.fullmatch(s):
            date = s
            continue

    return pd.Series([score, level, date])

df[["MSB_SCORE", "MSB_LEVEL", "MSB_SCORE_DATE"]] = df.apply(
    extract_msb_from_row,
    axis=1
)
