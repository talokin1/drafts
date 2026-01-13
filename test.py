with open("kved.json", encoding="utf-8") as f:
    kved_raw = json.load(f)

kved_new = pd.DataFrame([
    {
        "KVED_NEW": x.get("Код класу"),
        "KVED_NEW_NAME": x.get("Назва")
    }
    for x in kved_raw
    if "Код класу" in x and "Назва" in x
]).dropna()


def norm_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

df["DESCR_NORM"] = df["KVED_DESCR"].apply(norm_text)
kved_new["NAME_NORM"] = kved_new["KVED_NEW_NAME"].apply(norm_text)


def match_kved_descr(text, choices_df, threshold=85):
    if not text:
        return None, None, 0

    choices = choices_df["NAME_NORM"].tolist()

    match = process.extractOne(
        text,
        choices,
        scorer=fuzz.token_sort_ratio
    )

    if match is None:
        return None, None, 0

    matched_text, score, idx = match

    if score < threshold:
        return None, None, score

    row = choices_df.iloc[idx]
    return row["KVED_NEW"], row["KVED_NEW_NAME"], score

def get_division(kved):
    if not isinstance(kved, str):
        return None
    m = re.match(r"(\d{2})", kved)
    return m.group(1) if m else None
df["DIV"] = df["KVED"].apply(get_division)
kved_new["DIV"] = kved_new["KVED_NEW"].str[:2]



results = []

for i, row in df.iterrows():
    div = row["DIV"]
    text = row["DESCR_NORM"]

    pool = kved_new[kved_new["DIV"] == div] if div else kved_new

    new_kved, new_name, score = match_kved_descr(text, pool)

    results.append((new_kved, new_name, score))



df[["KVED_NEW", "KVED_NEW_NAME", "MATCH_SCORE"]] = pd.DataFrame(
    results, index=df.index
)
