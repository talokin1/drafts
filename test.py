845: [
    r"\bблагодійн(а|ої)\s+організаці(я|ї)\b",
    r"\bблагодійн(ий|ого)\s+фонд\b",
    r"\bбф\b",
],

825: [
    r"\bрелігійн(а|ої)\s+організаці(я|ї)\b",
    r"\bрелігійн(а|ої)\s+громад(а|и)\b",
    r"\bпарафі(я|ї)\b",
    r"\bхрам\b",
    r"\bцеркв(а|и)\b",
],



def extract_opf(full_name_norm: str):
    # ---------- LEVEL 1: exact ----------
    for _, r in opf_ref_exact.iterrows():
        if full_name_norm.startswith(r["OPF_NAME_NORM"]):
            return r["OPF_CODE"], r["OPF_NAME"], "EXACT"

    # ---------- LEVEL 2: top OPF text contain ----------
    for _, r in opf_ref_top.iterrows():
        if r["OPF_NAME_NORM"] in full_name_norm:
            return r["OPF_CODE"], r["OPF_NAME"], "TOP_TEXT"

    # ---------- LEVEL 3: marker-based (100% fallback) ----------
    for code, patterns in MAP_MARKERS_800.items():
        for pattern in patterns:
            if re.search(pattern, full_name_norm):
                name = (
                    opf_ref
                    .loc[opf_ref["OPF_CODE_INT"] == code, "OPF_NAME"]
                    .iloc[0]
                )
                return code, name, "MARKER"

    return None, None, "UNKNOWN"
