df.loc[mask, "acq_reason"] = (
    m_cp_inet.map({True: "cp_internet_acquiring", False: ""})
    + m_cp_keyword.map({True: "|cp_acquiring_keyword", False: ""})
).str.strip("|")


pp_norm = pp.map(normalize_ua)
m_refund_strong = pp_norm.str.contains(RE_REFUND_STRONG)
