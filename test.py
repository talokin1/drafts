import pandas as pd
import re
import numpy as np

df = df.copy()

LEVELS = {
    "дуже низький",
    "низький",
    "середній",
    "високий",
    "дуже високий",
}

DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")

def _to_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def _is_level(s: str) -> bool:
    return s.lower() in LEVELS

def _is_date(s: str) -> bool:
    return bool(DATE_RE.match(s))

def _is_score(s: str) -> bool:
    s = s.replace(",", ".")
    try:
        v = float(s)
        return 0 <= v <= 1000
    except:
        return False

def extract_msb_adjacent(row):
    vals = [_to_str(v) for v in row.values]
    low = [v.lower() for v in vals]

    idxs = [i for i, s in enumerate(low) if _is_level(s)]
    if not idxs:
        return pd.Series([np.nan, None, None])

    i = idxs[0]
    level = low[i]

    score = None
    date = None

    # базово: i-1 та i+1
    if i - 1 >= 0 and _is_score(vals[i - 1]):
        score = float(vals[i - 1].replace(",", "."))
    if i + 1 < len(vals) and _is_date(vals[i + 1]):
        date = vals[i + 1]

    # фолбек: маленьке вікно навколо рівня (на випадок зсуву на 1-2 клітинки)
    if score is None:
        for j in range(max(0, i - 3), min(len(vals), i + 4)):
            if j == i:
                continue
            if _is_score(vals[j]):
                score = float(vals[j].replace(",", "."))
                break

    if date is None:
        for j in range(max(0, i - 3), min(len(vals), i + 4)):
            if j == i:
                continue
            if _is_date(vals[j]):
                date = vals[j]
                break

    return pd.Series([score, level, date])

df[["MSB_SCORE", "MSB_LEVEL", "MSB_SCORE_DATE"]] = df.apply(extract_msb_adjacent, axis=1)
