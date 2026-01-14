opf_ref = (
    opf_df[["FIRM_OPFCD", "FIRM_OPFNM"]]
    .dropna()
    .drop_duplicates()
    .rename(columns={
        "FIRM_OPFCD": "OPF_CODE",
        "FIRM_OPFNM": "OPF_NAME"
    })
)

opf_ref["OPF_NAME_NORM"] = (
    opf_ref["OPF_NAME"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


df["FULL_NAME_NORM"] = (
    df["Повна назва"]
    .astype(str)
    .str.lower()
    .str.replace(r"[«»\"']", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

def extract_opf(full_name_norm, opf_ref):
    for _, r in opf_ref.iterrows():
        if full_name_norm.startswith(r["OPF_NAME_NORM"]):
            return r["OPF_CODE"], r["OPF_NAME"]
    return None, None
df[["OPF_CODE", "OPF_NAME"]] = df["FULL_NAME_NORM"].apply(
    lambda x: pd.Series(extract_opf(x, opf_ref))
)
