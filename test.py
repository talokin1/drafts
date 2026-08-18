import pandas as pd

parts = []

for col in [c for c in df.columns if c.startswith("COMPANY_")]:
    n = col.split("_")[-1]
    part = df[["PERSON_NAME", col, f"ROLE_{n}"]].rename(columns={col: "COMPANY", f"ROLE_{n}": "ROLE"}).dropna(subset=["COMPANY"])
    part["IDENTIFYCODE"] = part["COMPANY"].str.extract(r"\[(\d+)\]\s*$")[0]
    part["NAME_COMPANY"] = part["COMPANY"].str.replace(r"\s*\[\d+\]\s*$", "", regex=True).str.strip()
    parts.append(part[["PERSON_NAME", "NAME_COMPANY", "IDENTIFYCODE", "ROLE"]])

result = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)