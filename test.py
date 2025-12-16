import pandas as pd
import ast
import re

MAX_KVEDS = 5
KVED_COL = "kveds"

def parse_kveds(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    if isinstance(cell, list):
        raw = " ".join(cell)
    elif isinstance(cell, str):
        raw = cell.strip()
    else:
        return []

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            raw = " ".join(parsed)
    except Exception:
        pass

    raw = raw.replace("Код КВЕД", "").strip()
    parts = [p.strip() for p in raw.split(";") if p.strip()]

    result = []
    for p in parts:
        m = re.match(r"([\d.]+)\s+(.*)", p)
        if m:
            result.append((m.group(1), m.group(2)))

    return result

def expand_kveds(cell):
    data = parse_kveds(cell)
    out = {}

    for i in range(MAX_KVEDS):
        if i < len(data):
            code, descr = data[i]
            if i == 0:
                out["KVED"] = code
                out["KVED_DESCR"] = descr
            else:
                out[f"KVED_{i+1}"] = code
                out[f"KVED_{i+1}_DESCR"] = descr
        else:
            if i == 0:
                out["KVED"] = None
                out["KVED_DESCR"] = None
            else:
                out[f"KVED_{i+1}"] = None
                out[f"KVED_{i+1}_DESCR"] = None

    return pd.Series(out)

expanded = result[KVED_COL].apply(expand_kveds)
result = pd.concat([result.drop(columns=[KVED_COL]), expanded], axis=1)
result
