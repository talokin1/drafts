import ast
import pandas as pd

COL_AUTH = "Уповноважені особи"
COL_FOUN = "Засновники"
COL_BEN  = "Бенефіціари"

def to_list_of_dicts(x):
    if pd.isna(x) or x in ("", "[]"):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x.startswith("["):
            return []  # типу "Діє на підставі..." — не наш формат
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []

def classify(lst):
    """Повертає 'auth'/'found'/'ben'/'unknown' по ключах."""
    if not lst:
        return "unknown"
    keys = set()
    for d in lst:
        if isinstance(d, dict):
            keys |= set(d.keys())

    if ("Роль" in keys) and ("ПІБ" in keys):
        return "auth"
    if ("ПІБ / Назва" in keys) or ("Розмір внеску" in keys):
        return "found"
    if ("Тип володіння" in keys) or ("Частка" in keys):
        return "ben"

    # fallback-и (на випадок урізаних структур)
    if "Роль" in keys:
        return "auth"
    if "ПІБ / Назва" in keys:
        return "found"
    if "ПІБ" in keys and ("Країна" in keys):
        return "ben"

    return "unknown"

def repair_row(row):
    a = to_list_of_dicts(row.get(COL_AUTH))
    f = to_list_of_dicts(row.get(COL_FOUN))
    b = to_list_of_dicts(row.get(COL_BEN))

    buckets = {"auth": [], "found": [], "ben": []}

    for lst in (a, f, b):
        t = classify(lst)
        if t in buckets and lst:
            # якщо раптом два списки одного типу — склеїмо
            buckets[t].extend(lst)

    row[COL_AUTH] = buckets["auth"]
    row[COL_FOUN] = buckets["found"]
    row[COL_BEN]  = buckets["ben"]
    return row

df = df.apply(repair_row, axis=1)
