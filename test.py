TOP_OPF_KEYWORDS = [
    (800, ["профспілк", "профспілкова"]),
    (845, ["благодійн", "благодійний фонд"]),
    (825, ["релігійн", "релігійна громада", "парафія", "храм"]),
    (820, ["громадськ організаці", "громадська організація"]),
    (815, ["політична партія"]),
    (855, ["осбб", "обєднання співвласників"]),
    (860, ["орган самоорганізації населення"]),
    (300, ["кооператив"]),
]


def extract_opf(full_name_norm: str):
    # 1. Точний матч (нижній рівень)
    for _, r in opf_ref_exact.iterrows():
        if full_name_norm.startswith(r["OPF_NAME_NORM"]):
            return r["OPF_CODE"], r["OPF_NAME"]

    # 2. Верхній рівень через contains (рідко, але хай буде)
    for _, r in opf_ref_top.iterrows():
        if r["OPF_NAME_NORM"] in full_name_norm:
            return r["OPF_CODE"], r["OPF_NAME"]

    # 3. Keyword → top-level OPF
    for code, keys in TOP_OPF_KEYWORDS:
        if any(k in full_name_norm for k in keys):
            r = opf_ref_top.loc[opf_ref_top["OPF_CODE_INT"] == code]
            if not r.empty:
                row = r.iloc[0]
                return row["OPF_CODE"], row["OPF_NAME"]

    return None, None
