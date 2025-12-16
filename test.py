import pandas as pd
import ast
import re

# =========================
# CONFIG
# =========================
MAX_BENEFICIARIES = 5
BENEFICIARIES_COL = "beneficiaries"


# =========================
# SAFE PARSER
# =========================
def parse_beneficiaries(cell):
    """
    Повертає list[dict] або [].
    Працює з NaN, list, str.
    """

    # якщо вже список
    if isinstance(cell, list):
        return cell

    # None / NaN
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []

    # строка
    if isinstance(cell, str):
        s = cell.strip()
        if s == "" or s.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    return []


# =========================
# SHARE PARSER
# =========================
def parse_share(raw):
    """
    'відсоток частки - 30.0' -> 30.0
    """
    if not raw or not isinstance(raw, str):
        return None

    m = re.search(r"([\d.]+)", raw)
    return float(m.group(1)) if m else None


# =========================
# EXPAND FUNCTION
# =========================
def expand_beneficiaries(cell):
    data = parse_beneficiaries(cell)
    out = {}

    for i in range(MAX_BENEFICIARIES):
        person = data[i] if i < len(data) else None

        if isinstance(person, dict):
            out[f"beneficiary_{i+1}"] = person.get("ПІБ")
            out[f"share_{i+1}"] = parse_share(person.get("Частка"))
        else:
            out[f"beneficiary_{i+1}"] = None
            out[f"share_{i+1}"] = None

    return pd.Series(out)


# =========================
# MAIN TRANSFORM
# =========================
def split_beneficiaries_wide(df: pd.DataFrame) -> pd.DataFrame:
    expanded = df[BENEFICIARIES_COL].apply(expand_beneficiaries)
    df_out = pd.concat(
        [df.drop(columns=[BENEFICIARIES_COL]), expanded],
        axis=1
    )
    return df_out


# =========================
# USAGE
# =========================
# result = pd.read_csv("your_file.csv")
result = split_beneficiaries_wide(result)

result
