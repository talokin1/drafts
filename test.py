import pandas as pd

TABLES = [
    "csbm.IF_IFOBSSMSDELIVERY@DWH",
    "csbm.IF_PUSHMESSAGE@DWH",
    "csbm.IF_MAILSMSTYPE@DWH"
]

ID_COL = "USERID"
TEXT_COL = "TEXT"


def chunks(values, n=900):
    values = list(values)
    for i in range(0, len(values), n):
        yield values[i:i+n]


def sql_in(values):
    return ",".join(f"'{str(x).strip()}'" for x in values)


found_parts = []

for table in TABLES:
    for client_part in chunks(clients, 900):
        ids = sql_in(client_part)

        QUERY = f"""
            SELECT
                {ID_COL} AS CONTRAGENTID,
                {TEXT_COL} AS TEXT
            FROM {table}
            WHERE {ID_COL} IN ({ids})
        """

        temp = get_data(QUERY)

        if temp is not None and len(temp) > 0:
            temp["SOURCE_TABLE"] = table
            found_parts.append(temp)


found_df = pd.concat(found_parts, ignore_index=True)

found_df["CONTRAGENTID"] = (
    found_df["CONTRAGENTID"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

found_df = found_df.drop_duplicates()