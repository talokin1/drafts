clients_ids = clients["CONTRAGENTID"].astype(int).tolist()

dates_sql = [
    f"date '{pd.to_datetime(d, dayfirst=True).date()}'"
    for d in dates
]


QUERY = f"""
select
    contragentid,
    arcdate,
    max(balance_amt_uah) as max_balance_amt_uah
from b2_olap.ar_deals@dwh
where contragentid in ({','.join(map(str, clients_ids))})
  and arcdate in ({','.join(dates_sql)})
group by contragentid, arcdate
"""

df = get_data(QUERY)
result = (
    df
    .pivot(
        index="CONTRAGENTID",
        columns="ARCDATE",
        values="MAX_BALANCE_AMT_UAH"
    )
    .fillna(0)
    .sort_index()
)
result.columns = result.columns.strftime("%d.%m.%Y")
