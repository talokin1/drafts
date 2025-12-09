def find_columns_with_authorized(df):
    authorized_columns = ["Уповноважені особи"]  # базова колонка

    for col in df.columns:
        if col == "Уповноважені особи":
            continue  # вже додана

        for val in df[col]:
            parsed = safe_parse(val)

            # список словників
            if isinstance(parsed, list):
                if any(isinstance(item, dict) and "Роль" in item for item in parsed):
                    authorized_columns.append(col)
                    break

            # окремий словник
            elif isinstance(parsed, dict):
                if "Роль" in parsed:
                    authorized_columns.append(col)
                    break

    return authorized_columns


def extract_all_authorized(df, authorized_cols):
    authorized_list = []

    for idx, row in df.iterrows():
        combined = []

        for col in authorized_cols:
            parsed = safe_parse(row[col])

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "Роль" in item:
                        combined.append(item)

            elif isinstance(parsed, dict):
                if "Роль" in parsed:
                    combined.append(parsed)

        authorized_list.append(combined)

    df["Authorized"] = authorized_list
    return df

df["Authorized"].apply(lambda x: isinstance(x, list) and len(x)==0).sum()


def expand_authorized_column(df, source_col="Authorized", max_items=10):
    result = {}

    for idx, val in df[source_col].items():
        entry = {}

        if not isinstance(val, list):
            result[idx] = entry
            continue

        for i, person in enumerate(val[:max_items], start=1):
            if not isinstance(person, dict):
                continue

            entry[f"Authorized_{i}_Name"] = person.get("ПІБ")
            entry[f"Authorized_{i}_Role"] = person.get("Роль")

        result[idx] = entry

    return pd.DataFrame.from_dict(result, orient="index")

authorized_cols = find_columns_with_authorized(df)
print("Знайдені колонки з Уповноваженими особами:", authorized_cols)

df = extract_all_authorized(df, authorized_cols)

df_authorized_expanded = expand_authorized_column(df, "Authorized", max_items=10)

df = pd.concat([df, df_authorized_expanded], axis=1)
df


