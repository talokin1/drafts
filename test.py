import pandas as pd
import re
import unicodedata

# --- load OPF reference ---
opf_df = pd.read_excel(r"M:\Controlling\Data_Science_Projects\Financial_indicators_tool\OPFCD_OPFNM.xlsx")

opf_ref = (
    opf_df[["FIRM_OPFCD", "FIRM_OPFNM"]]
    .dropna()
    .drop_duplicates()
    .rename(columns={
        "FIRM_OPFCD": "OPF_CODE",
        "FIRM_OPFNM": "OPF_NAME"
    })
)

_APOS_RE = re.compile(r"[ʼ’`´']")
_DASH_RE = re.compile(r"[–—−]")

def norm_ua(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s).lower()
    s = _APOS_RE.sub("", s)
    s = _DASH_RE.sub("-", s)
    s = re.sub(r"[«»\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

opf_ref["OPF_NAME_NORM"] = opf_ref["OPF_NAME"].map(norm_ua)
opf_ref["OPF_CODE_INT"] = opf_ref["OPF_CODE"].astype(int)

opf_ref_exact = (
    opf_ref
    .sort_values("OPF_NAME_NORM", key=lambda s: s.str.len(), ascending=False)
    .reset_index(drop=True)
)

opf_ref_top = (
    opf_ref
    .query("OPF_CODE_INT % 100 == 0")
    .sort_values("OPF_NAME_NORM", key=lambda s: s.str.len(), ascending=False)
    .reset_index(drop=True)
)

def extract_opf(full_name_norm: str):
    for _, r in opf_ref_exact.iterrows():
        if full_name_norm.startswith(r["OPF_NAME_NORM"]):
            return r["OPF_CODE"], r["OPF_NAME"]

    for _, r in opf_ref_top.iterrows():
        if r["OPF_NAME_NORM"] in full_name_norm:
            return r["OPF_CODE"], r["OPF_NAME"]

    return None, None

# --- apply to dataframe df ---
df["FULL_NAME_NORM"] = df["FULL_NAME"].map(norm_ua)

df[["OPF_CODE", "OPF_NAME"]] = df["FULL_NAME_NORM"].apply(
    lambda x: pd.Series(extract_opf(x))
)
