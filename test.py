import pandas as pd
import ast

# =========================
# CONFIG
# =========================
MAX_PERSONS = 5
AUTHORIZED_COL = "authorized"
ID_COL = "IDENTIFYCODE"


# =========================
# SAFE PARSER
# =========================
def parse_authorized(cell):
    """
    Повертає list[dict] або [].
    Безпечно для NaN, list, str, будь-чого.
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

    # все інше
    return []


# =========================
# EXPAND FUNCTION
# =========================
def expand_authorized(cell):
    data = parse_authorized(cell)
    out = {}

    for i in range(MAX_PERSONS):
        person = data[i] if i < len(data) else None

        if isinstance(person, dict):
            out[f"pib_{i+1}"] = person.get("ПІБ")
            out[f"role_{i+1}"] = person.get("Роль")
        else:
            out[f"pib_{i+1}"] = None
            out[f"role_{i+1}"] = None

    return pd.Series(out)


# =========================
# MAIN TRANSFORM
# =========================
def split_authorized_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляє колонку `authorized`
    і додає pib_1 / role_1 ... pib_5 / role_5
    """
    expanded = df[AUTHORIZED_COL].apply(expand_authorized)
    df_out = pd.concat(
        [df.drop(columns=[AUTHORIZED_COL]), expanded],
        axis=1
    )
    return df_out


# =========================
# USAGE
# =========================
# result = pd.read_csv("your_file.csv")   # якщо треба
result = split_authorized_wide(result)

result
