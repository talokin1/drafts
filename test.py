import pandas as pd

records = []

for idx, row in clean_df.iterrows():
    company_id = row.get("ЄДРПОУ") or idx  # якщо ЄДРПОУ вже є — використовуй його

    for person in row["authorized"]:
        records.append({
            "company_id": company_id,
            "full_name": person.get("ПІБ"),
            "role_raw": person.get("Роль")
        })

authorized_df = pd.DataFrame(records)
