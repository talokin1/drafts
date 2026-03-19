ids = df["IDENTIFYCODE"].dropna().astype(str).unique().tolist()

def chunker(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


result_dfs = []

for chunk in chunker(ids, 1000):
    ids_str = ", ".join([f"'{x}'" for x in chunk])

    query = f"""
    select ID as CONTRAGENTID, IDENTIFYCODE
    from b2_olap.ar_contragent@dwh
    where arcdate = '18.03.2026'
      and IDENTIFYCODE in ({ids_str})
      and closedate is null
    """

    temp_df = get_data(query)
    result_dfs.append(temp_df)

contragents = pd.concat(result_dfs, ignore_index=True)

df = df.merge(
    contragents,
    on="IDENTIFYCODE",
    how="left"
)