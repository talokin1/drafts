import pdfplumber
import pandas as pd

rows = []

with pdfplumber.open(r"C:\Projects\(DS-248) Parsing YouControl\uBKI_v2\kved20101-2005.cleaned.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if not table:
            continue
        rows.extend(table)

df = pd.DataFrame(
    rows,
    columns=[
        "kved_2010",
        "name_2010",
        "kved_2005",
        "name_2005",
        "description"
    ]
)

df = df[~df["kved_2010"].str.contains("КВЕД", na=False)]
df = df.dropna(how="all")

df[["kved_2010", "name_2010"]] = df[["kved_2010", "name_2010"]].ffill()
df[["kved_2005", "name_2005"]] = df[["kved_2005", "name_2005"]].ffill()

def clean_code(x):
    if not isinstance(x, str):
        return None
    x = x.strip()
    return x if x else None

for c in ["kved_2010", "kved_2005"]:
    df[c] = df[c].apply(clean_code)

mapping = (
    df[["kved_2005", "kved_2010", "name_2010"]]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)

mapping.to_csv(
    "kved_2005_to_2010.csv",
    index=False,
    encoding="utf-8"
)

mapping["kved_2005"].nunique()
mapping["kved_2010"].nunique()
mapping.head(20)
