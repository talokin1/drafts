results = []
MAX_IN = 900  # запас від 1000

for arcdate, group in data.groupby("Дата проведення"):
    ids = (
        group["Ідентифікатор заявки в АБС"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(ids) == 0:
        continue

    for i in range(0, len(ids), MAX_IN):
        sub_ids = ids[i:i + MAX_IN]
        ids_str = ",".join(map(str, sub_ids))

        sql = f"""
        SELECT
            arcdate,
            id,
            bankaID,
            bankbID
        FROM b2_olap.ar_document@dwh
        WHERE arcdate = TO_DATE('{arcdate.strftime("%d.%m.%Y")}', 'DD.MM.YYYY')
          AND id IN ({ids_str})
        """

        df_part = get_data(sql)
        results.append(df_part)

    print(f"{arcdate.date()} → {len(ids)} IDs")
