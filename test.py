import pandas as pd

dfs = [t.df for t in tables]
raw = pd.concat(dfs, ignore_index=True)
raw.head()

raw.columns = [
    "kved_2010",
    "name_2010",
    "kved_2005",
    "name_2005",
    "description"
]

raw = raw.iloc[1:]  # прибрати повторні заголовки
raw = raw.rename(columns={
    0: "kved_2010",
    1: "name_2010",
    2: "kved_2005",
    3: "name_2005",
    4: "description"
})

def clean_code(x):
    if not isinstance(x, str):
        return None
    x = x.strip()
    return x if x else None

for c in ["kved_2010", "kved_2005"]:
    raw[c] = raw[c].apply(clean_code)

raw["kved_2010"] = raw["kved_2010"].replace("", None)
raw["kved_2005"] = raw["kved_2005"].replace("", None)

raw[["kved_2010", "name_2010"]] = raw[["kved_2010", "name_2010"]].ffill()
raw[["kved_2005", "name_2005"]] = raw[["kved_2005", "name_2005"]].ffill()

mapping = raw[[
    "kved_2005",
    "kved_2010",
    "name_2010"
]].dropna()

mapping.head(20)
mapping["kved_2005"].value_counts().head()



mapping.to_csv(
    "kved_2005_to_2010.csv",
    index=False,
    encoding="utf-8"
)

