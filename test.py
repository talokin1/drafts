def extract_opf(full_name_norm: str):
    # ---------- LEVEL 1: exact ----------
    for _, r in opf_ref_exact.iterrows():
        if full_name_norm.startswith(r["OPF_NAME_NORM"]):
            return r["OPF_CODE"], r["OPF_NAME"], "EXACT"

    # ---------- LEVEL 2: top OPF text contain ----------
    for _, r in opf_ref_top.iterrows():
        if r["OPF_NAME_NORM"] in full_name_norm:
            return r["OPF_CODE"], r["OPF_NAME"], "TOP_TEXT"

    # ---------- LEVEL 3: lemma-based TOP ----------
    lemmas = lemmatize_ua(full_name_norm)
    top_code = match_top_opf(lemmas)

    if top_code:
        name = opf_top.loc[opf_top["OPF_CODE"] == top_code, "OPF_NAME"].iloc[0]
        return top_code, name, "TOP_LEMMA"

    return None, None, "UNKNOWN"


df[["OPF_CODE", "OPF_NAME", "OPF_SOURCE"]] = (
    df["FULL_NAME_NORM"]
    .apply(lambda x: pd.Series(extract_opf(x)))
)
