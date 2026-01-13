import re

def norm(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


df["DESCR_NORM"] = df["KVED_DESCR"].apply(norm)
kved_new["NAME_NORM"] = kved_new["KVED_NEW_NAME"].apply(norm)


STOPWORDS = {
    "діяльність", "та", "у", "і", "з", "на", "по", "інших", "інша",
    "організацій", "організації", "професійних"
}

def anchors(name):
    return [
        w for w in name.split()
        if len(w) > 4 and w not in STOPWORDS
    ]
kved_new["ANCHORS"] = kved_new["NAME_NORM"].apply(anchors)


def match_by_in(text, ref_df):
    if not text:
        return None, None

    for _, r in ref_df.iterrows():
        if not r["ANCHORS"]:
            continue

        if all(a in text for a in r["ANCHORS"]):
            return r["KVED_NEW"], r["KVED_NEW_NAME"]

    return None, None

df["DIV"] = df["KVED"].str[:2]
kved_new["DIV"] = kved_new["KVED_NEW"].str[:2]



res = []

for _, row in df.iterrows():
    pool = kved_new[kved_new["DIV"] == row["DIV"]]
    kved_new_code, kved_new_name = match_by_in(
        row["DESCR_NORM"],
        pool
    )
    res.append((kved_new_code, kved_new_name))
df[["KVED_NEW", "KVED_NEW_NAME"]] = res
