import pandas as pd
import ast

# =========================
# CONFIG
# =========================
MAX_FOUNDERS = 5
FOUNDERS_COL = "founders"


# =========================
# SAFE PARSER
# =========================
def parse_founders(cell):
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
# EXPAND FUNCTION
# =========================
def expand_founders(cell):
    data = parse_founders(cell)
    out = {}

    for i in range(MAX_FOUNDERS):
        person = data[i] if i < len(data) else None

        if isinstance(person, dict):
            out[f"founder_{i+1}"] = person.get("ПІБ / Назва")
        else:
            out[f"founder_{i+1}"] = None

    return pd.Series(out)


# =========================
# MAIN TRANSFORM
# =========================
def split_founders_wide(df: pd.DataFrame) -> pd.DataFrame:
    expanded = df[FOUNDERS_COL].apply(expand_founders)
    df_out = pd.concat(
        [df.drop(columns=[FOUNDERS_COL]), expanded],
        axis=1
    )
    return df_out


# =========================
# USAGE
# =========================
# result = pd.read_csv("your_file.csv")
result = split_founders_wide(result)

result
