
data["Дата проведення"] = pd.to_datetime(
    data["Дата проведення"],
    dayfirst=True
)
data = data.sort_values("Дата проведення")




results = []

for arcdate, group in data.groupby("Дата проведення"):
    ids = group["Ідентифікатор заявки в АБС"].dropna().unique()

    if len(ids) == 0:
        continue


    ids_str = ",".join(str(int(x)) for x in ids)

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

bank_df = pd.concat(results, ignore_index=True)


data = data.merge(
    bank_df,
    left_on=["Дата проведення", "Ідентифікатор заявки в АБС"],
    right_on=["arcdate", "id"],
    how="left"
)
