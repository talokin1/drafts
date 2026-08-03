import re
import pandas as pd

ID_COL, COMPANY_COL = "IDENTIFYCODE", "FIRM_NAME"
MAX_MATCHED_COMPANIES, MAX_OUTPUT_COMPANIES = 20, 10

STOP_WORDS = {
    "ТОВ", "ПП", "ПАТ", "ПРАТ", "АТ", "ФГ", "КП", "ДП", "ГО", "ОСББ",
    "ТОВАРИСТВО", "ПІДПРИЄМСТВО", "КОМПАНІЯ", "ФІРМА", "КООПЕРАТИВ",
    "ГОСПОДАРСТВО", "СПІВВЛАСНИКИ", "ВЛАСНИКИ", "ЗАСНОВНИКИ",
    "УЧАСНИКИ", "ЧЛЕНИ", "АКЦІОНЕРИ", "БЕНЕФІЦІАР", "ВІДСУТНІЙ",
    "ВІДСУТНЯ", "НЕВІДОМО", "ІНФОРМАЦІЯ"
}

PATRONYMIC_PATTERN = r"(ОВИЧ|ЕВИЧ|ЄВИЧ|ЙОВИЧ|ИЧ|ІЧ|ОВНА|ЕВНА|ЄВНА|ИВНА|ІВНА|ЇВНА)$"


def normalize_pib(value):
    if pd.isna(value): return pd.NA

    value = str(value).upper().replace("Ё", "Е").replace("’", "'").replace("`", "'").replace("ʼ", "'")
    value = re.sub(r"\s+", " ", value).strip()

    if not re.fullmatch(r"[А-ЯІЇЄҐ'\-\s]+", value): return pd.NA

    words = value.split()

    if len(words) != 3: return pd.NA
    if any(len(word.replace("-", "").replace("'", "")) < 2 for word in words): return pd.NA
    if any(word in STOP_WORDS for word in words): return pd.NA
    if not re.search(PATRONYMIC_PATTERN, words[-1]): return pd.NA

    return value


def find_person_matches(dataset):
    specs = [
        ("BENEFICIARY_NAME_", None, "Бенефіціар"),
        ("FOUNDER_NAME_", None, "Засновник"),
        ("AUTHORISED_NAME_", "AUTHORISED_ROLE_", "Уповноважена особа")
    ]

    parts = []

    for name_prefix, role_prefix, default_role in specs:
        for i in range(1, 6):
            name_col = f"{name_prefix}{i}"
            role_col = f"{role_prefix}{i}" if role_prefix else None

            if name_col not in dataset.columns: continue

            columns = [ID_COL, COMPANY_COL, name_col]
            if role_col in dataset.columns: columns.append(role_col)

            part = dataset[columns].drop_duplicates().rename(columns={name_col: "PERSON_NAME"})

            if role_col in part.columns:
                part = part.rename(columns={role_col: "ROLE"})
                part["ROLE"] = part["ROLE"].astype("string").str.strip().replace("", pd.NA).fillna(default_role)
            else:
                part["ROLE"] = default_role

            parts.append(part)

    links = pd.concat(parts, ignore_index=True)
    links[ID_COL] = links[ID_COL].astype("string").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(8)
    links["PERSON_NAME"] = links["PERSON_NAME"].map(normalize_pib)
    links = links.dropna(subset=["PERSON_NAME", ID_COL]).drop_duplicates(["PERSON_NAME", ID_COL, "ROLE"])

    links = links.groupby(["PERSON_NAME", ID_COL, COMPANY_COL], as_index=False, dropna=False)["ROLE"].agg(lambda x: " | ".join(sorted(set(x))))

    stats = links.groupby("PERSON_NAME").agg(TOTAL_COMPANIES=(ID_COL, "nunique"), HAS_STRONG_ROLE=("ROLE", lambda x: x.str.contains("Бенефіціар|Засновник", regex=True).any())).reset_index()

    links = links.merge(stats, on="PERSON_NAME")
    links = links[links["TOTAL_COMPANIES"].between(2, MAX_MATCHED_COMPANIES) & links["HAS_STRONG_ROLE"]].copy()

    links["ROLE_PRIORITY"] = links["ROLE"].str.contains("Бенефіціар").astype(int) * 2 + links["ROLE"].str.contains("Засновник").astype(int)
    links = links.sort_values(["PERSON_NAME", "ROLE_PRIORITY", COMPANY_COL], ascending=[True, False, True])
    links["N"] = links.groupby("PERSON_NAME").cumcount().add(1)

    limited = links[links["N"].le(MAX_OUTPUT_COMPANIES)].copy()
    limited["COMPANY"] = limited[COMPANY_COL].fillna("Без назви") + " [" + limited[ID_COL] + "]"

    companies = limited.pivot(index="PERSON_NAME", columns="N", values="COMPANY").add_prefix("COMPANY_")
    roles = limited.pivot(index="PERSON_NAME", columns="N", values="ROLE").add_prefix("ROLE_")
    wide = pd.concat([companies, roles], axis=1)

    counts = links.groupby("PERSON_NAME")[ID_COL].nunique()
    wide.insert(0, "TOTAL_COMPANIES", counts)
    wide.insert(1, "IS_TRUNCATED", wide["TOTAL_COMPANIES"].gt(MAX_OUTPUT_COMPANIES))

    ordered = [column for i in range(1, MAX_OUTPUT_COMPANIES + 1) for column in (f"COMPANY_{i}", f"ROLE_{i}") if column in wide.columns]
    wide = wide.reset_index()[["PERSON_NAME", "TOTAL_COMPANIES", "IS_TRUNCATED"] + ordered]

    links_long = links.drop(columns=["HAS_STRONG_ROLE", "ROLE_PRIORITY", "N"]).sort_values(["PERSON_NAME", COMPANY_COL])

    return wide, links_long

matches_df, matches_long = find_person_matches(dataset)