import re
import pandas as pd
import networkx as nx

ID_COL = "IDENTIFYCODE"
NAME_COL = "FIRM_NAME"

ROLE_COLUMNS = {
    "BENEFICIARY": [f"BENEFICIARY_NAME_{i}" for i in range(1, 6)],
    "FOUNDER": [f"FOUNDER_NAME_{i}" for i in range(1, 6)]
}

LEGAL_ENTITY_WORDS = r"\b(ТОВ|ПП|ПАТ|ПРАТ|АТ|ФГ|КП|ДП|КОМПАНІЯ|ПІДПРИЄМСТВО|ГОСПОДАРСТВО|LIMITED|LLC|LTD)\b"


def normalize_name(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).upper().strip()
    value = value.replace("Ё", "Е").replace("Ґ", "Г").replace("’", "'").replace("`", "'").replace("ʼ", "'")
    value = re.sub(r"[^А-ЯІЇЄA-Z'\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if len(value.split()) >= 2 else pd.NA


def find_business_groups(dataset, include_founders=True):
    roles = ["BENEFICIARY", "FOUNDER"] if include_founders else ["BENEFICIARY"]
    companies = dataset[[ID_COL, NAME_COL]].drop_duplicates(ID_COL).copy()
    links = []

    for role in roles:
        columns = [col for col in ROLE_COLUMNS[role] if col in dataset.columns]

        for col in columns:
            part = dataset[[ID_COL, NAME_COL, col]].rename(columns={col: "PERSON_NAME"}).copy()
            part["ROLE"] = role
            links.append(part)

    links = pd.concat(links, ignore_index=True)
    links["PERSON_KEY"] = links["PERSON_NAME"].map(normalize_name)
    links = links.dropna(subset=[ID_COL, "PERSON_KEY"])
    links = links[~links["PERSON_KEY"].str.contains(LEGAL_ENTITY_WORDS, regex=True, na=False)]
    links = links.drop_duplicates([ID_COL, "PERSON_KEY", "ROLE"])

    person_counts = links.groupby("PERSON_KEY")[ID_COL].nunique()
    links = links[links["PERSON_KEY"].isin(person_counts[person_counts >= 2].index)]

    graph = nx.Graph()

    for row in links.itertuples(index=False):
        graph.add_edge(f"COMPANY:{getattr(row, ID_COL)}", f"PERSON:{row.PERSON_KEY}")

    group_map = {}

    for group_number, component in enumerate(nx.connected_components(graph), start=1):
        company_ids = [node.replace("COMPANY:", "") for node in component if node.startswith("COMPANY:")]

        if len(company_ids) < 2:
            continue

        group_id = f"BG_{group_number:06d}"

        for company_id in company_ids:
            group_map[company_id] = group_id

    companies["BUSINESS_GROUP_ID"] = companies[ID_COL].map(group_map)

    result = links.merge(companies[[ID_COL, "BUSINESS_GROUP_ID"]], on=ID_COL, how="left")
    result = result.dropna(subset=["BUSINESS_GROUP_ID"])

    group_summary = (
        result.groupby("BUSINESS_GROUP_ID")
        .agg(
            GROUP_SIZE=(ID_COL, "nunique"),
            COMPANIES=(NAME_COL, lambda x: " | ".join(sorted(set(x.dropna())))),
            RELATED_PERSONS=("PERSON_KEY", lambda x: " | ".join(sorted(set(x)))),
            CONNECTION_ROLES=("ROLE", lambda x: " | ".join(sorted(set(x))))
        )
        .reset_index()
        .sort_values("GROUP_SIZE", ascending=False)
    )

    company_groups = (
        result.groupby(["BUSINESS_GROUP_ID", ID_COL, NAME_COL], as_index=False)
        .agg(
            MATCHED_PERSONS=("PERSON_KEY", lambda x: " | ".join(sorted(set(x)))),
            MATCHED_ROLES=("ROLE", lambda x: " | ".join(sorted(set(x))))
        )
    )

    return company_groups, group_summary, links


company_groups, group_summary, person_links = find_business_groups(dataset)