if s.startswith("["):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            keys = set().union(*(d.keys() for d in parsed if isinstance(d, dict)))
            if "Код" in keys and "Назва" in keys:
                return "kved_list", parsed
            if "ПІБ" in keys:
                return "people_list", parsed
        return "noise", s
    except Exception:
        return "noise", s


elif kind == "kved_list":
    for d in val:
        if isinstance(d, dict) and "Код" in d:
            out["kveds"].append(d["Код"])



clean_df["KVED"] = clean_df["kveds"].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
)
