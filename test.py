with open("kved.json", "r", encoding="utf-8") as f:
    data = json.load(f)

kved_map = pd.DataFrame(data)

kved_map = kved_map.rename(columns={
    "Код секції": "SECTION",
    "Код розділу \n": "DIVISION",
    "Код групи \n": "GROUP",
    "Код класу": "CLASS",
    "Назва": "NAME"
})

def norm(x):
    if not isinstance(x, str):
        return ""
    x = x.lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()
df["DESCR_NORM"] = df["KVED_DESCR"].apply(norm)
kved_map["NAME_NORM"] = kved_map["NAME"].apply(norm)


def match_meta(descr):
    if not descr:
        return None, None, None, None

    for _, row in kved_map.iterrows():
        if row["NAME_NORM"] in descr or descr in row["NAME_NORM"]:
            return (
                row["SECTION"],
                row["DIVISION"],
                row["GROUP"],
                row["CLASS"],
            )

    return None, None, None, None


df[["SECTION", "DIVISION", "GROUP", "CLASS"]] = (
    df["DESCR_NORM"]
    .apply(match_meta)
    .apply(pd.Series)
)

df.drop(columns=["DESCR_NORM"], inplace=True)
