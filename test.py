import pandas as pd
import ast

def parse_authorized(cell):
    if pd.isna(cell) or cell.strip() == "":
        return []
    try:
        return ast.literal_eval(cell)
    except Exception:
        return []

df["authorized_parsed"] = df["authorized"].apply(parse_authorized)

rows = []

for _, row in df.iterrows():
    company_id = row["ID"]
    for i, person in enumerate(row["authorized_parsed"], start=1):
        rows.append({
            "ID": company_id,
            "person_idx": i,
            "pib": person.get("ПІБ"),
            "role": person.get("Роль")
        })

authorized_df = pd.DataFrame(rows)

MAX_PERSONS = 5

def expand_authorized(cell):
    data = parse_authorized(cell)
    out = {}
    for i in range(MAX_PERSONS):
        if i < len(data):
            out[f"pib_{i+1}"] = data[i].get("ПІБ")
            out[f"role_{i+1}"] = data[i].get("Роль")
        else:
            out[f"pib_{i+1}"] = None
            out[f"role_{i+1}"] = None
    return pd.Series(out)

expanded = df["authorized"].apply(expand_authorized)
df = pd.concat([df.drop(columns=["authorized"]), expanded], axis=1)
