buckets = {
    "authorized": [],
    "founders": [],
    "beneficiaries": [],
    "kveds": [],
    "bankruptcy": None,
    "tax": {},
    "finance": {},
    "msb": {},
    "meta": []
}


import ast, re, pandas as pd

def parse_money(x):
    if not isinstance(x, str):
        return None
    m = re.search(r"([\d\s.,]+)\s*тис\.грн", x)
    if m:
        return float(m.group(1).replace(" ", "").replace(",", ".")) * 1000
    return None

def detect_cell(x):
    if pd.isna(x):
        return "empty", None

    if isinstance(x, str):
        s = x.strip()

        # списки людей
        if s.startswith("[") and "ПІБ" in s:
            try:
                return "list", ast.literal_eval(s)
            except:
                return "noise", s

        if "Код КВЕД" in s:
            return "kved", s

        if "Немає інформації про банкрутство" in s:
            return "bankruptcy", s

        if "Податкові дані" in s:
            return "tax_block", s

        if "Не є платником ПДВ" in s or "платником ПДВ" in s:
            return "vat", s

        if "тис.грн" in s:
            return "money", s

        if re.match(r"\d{2}\.\d{2}\.\d{4}", s):
            return "date", s

        if s.isdigit():
            return "number", int(s)

        return "text", s

    return "unknown", x


def rebuild_row(row):
    out = {
        "authorized": [],
        "founders": [],
        "beneficiaries": [],
        "kveds": [],
        "bankruptcy": None,
        "vat_status": None,
        "income": None,
        "net_profit": None,
        "assets": None,
        "liabilities": None,
        "salary_debt": None,
        "msb_score": None,
        "msb_level": None,
        "meta": []
    }

    for cell in row.values:
        kind, val = detect_cell(cell)

        if kind == "list":
            keys = set().union(*(d.keys() for d in val if isinstance(d, dict)))
            if "Роль" in keys:
                out["authorized"].extend(val)
            elif "ПІБ / Назва" in keys:
                out["founders"].extend(val)
            elif "Тип володіння" in keys or "Частка" in keys:
                out["beneficiaries"].extend(val)

        elif kind == "kved":
            out["kveds"].append(val)

        elif kind == "bankruptcy":
            out["bankruptcy"] = val

        elif kind == "vat":
            out["vat_status"] = val

        elif kind == "money":
            m = parse_money(val)
            if out["income"] is None:
                out["income"] = m
            elif out["assets"] is None:
                out["assets"] = m
            elif out["liabilities"] is None:
                out["liabilities"] = m
            else:
                out["salary_debt"] = m

        elif kind == "text":
            out["meta"].append(val)

        elif kind == "number":
            if out["msb_level"] is None:
                out["msb_level"] = val

    return out

clean_df = pd.DataFrame(df.apply(rebuild_row, axis=1).tolist())
