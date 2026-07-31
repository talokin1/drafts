import re
import unicodedata
import pandas as pd

LEGAL_MARKERS = re.compile(r"\b(ТОВАРИСТВО|ПІДПРИЄМСТВО|КОМПАНІЯ|ФЕРМЕРСЬКЕ|ГОСПОДАРСТВО|ТОВ|ПП|ПАТ|ПРАТ|АТ|ФГ|КП|ДП|LLC|LTD)\b")


def normalize_pib(value):
    if pd.isna(value): return pd.NA

    value = unicodedata.normalize("NFKC", str(value)).upper()
    value = value.replace("Ё", "Е").replace("’", "'").replace("`", "'").replace("ʼ", "'")
    value = re.sub(r"[^А-ЯІЇЄҐA-Z'\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    if len(value.split()) < 2 or LEGAL_MARKERS.search(value): return pd.NA
    return value


def find_person_company_matches(dataset):
    id_col, company_col = "IDENTIFYCODE", "FIRM_NAME"

    specifications = [
        ("BENEFICIARY_NAME_", None, "Бенефіціар"),
        ("FOUNDER_NAME_", None, "Засновник"),
        ("AUTHORISED_NAME_", "AUTHORISED_ROLE_", "Уповноважена особа")
    ]

    parts = []

    for name_prefix, role_prefix, default_role in specifications:
        for i in range(1, 6):
            name_col = f"{name_prefix}{i}"
            role_col = f"{role_prefix}{i}" if role_prefix else None

            if name_col not in dataset.columns: continue

            selected = [id_col, company_col, name_col]
            if role_col in dataset.columns: selected.append(role_col)

            part = dataset[selected].rename(columns={name_col: "PERSON_NAME"}).copy()

            if role_col in part.columns:
                part = part.rename(columns={role_col: "ROLE"})
                part["ROLE"] = part["ROLE"].astype("string").str.strip().replace("", pd.NA).fillna(default_role)
            else:
                part["ROLE"] = default_role

            parts.append(part)

    links = pd.concat(parts, ignore_index=True)
    links[id_col] = links[id_col].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    links["PERSON_NAME"] = links["PERSON_NAME"].map(normalize_pib)
    links = links.dropna(subset=[id_col, "PERSON_NAME"])
    links = links.drop_duplicates(["PERSON_NAME", id_col, "ROLE"])

    links = links.groupby(["PERSON_NAME", id_col, company_col], as_index=False)["ROLE"].agg(lambda x: " | ".join(sorted(set(x))))
    links = links[links.groupby("PERSON_NAME")[id_col].transform("nunique").ge(2)].copy()

    if links.empty: return pd.DataFrame(columns=["PERSON_NAME"])

    links["COMPANY"] = links[company_col].fillna("Без назви") + " [" + links[id_col] + "]"
    links = links.sort_values(["PERSON_NAME", "COMPANY"])
    links["N"] = links.groupby("PERSON_NAME").cumcount().add(1)

    companies = links.pivot(index="PERSON_NAME", columns="N", values="COMPANY").add_prefix("COMPANY_")
    roles = links.pivot(index="PERSON_NAME", columns="N", values="ROLE").add_prefix("ROLE_")

    result = pd.concat([companies, roles], axis=1)
    result.columns.name = None

    max_n = int(links["N"].max())
    ordered_columns = [column for i in range(1, max_n + 1) for column in (f"COMPANY_{i}", f"ROLE_{i}")]

    return result.reset_index()[["PERSON_NAME"] + ordered_columns]

matches_df = find_person_company_matches(dataset)