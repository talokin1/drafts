base_clients = (
    result
    .loc[:, ["IDENTIFYCODE", "REGISTERDATE"]]
    .drop_duplicates()
)

base_clients["REGISTERDATE"] = pd.to_datetime(base_clients["REGISTERDATE"]).dt.date

arcdates = sorted(base_clients["REGISTERDATE"].unique())


SELECT 
    arcdate,
    CONTRAGENTID_ZP,
    CONTRAGENTID_EMPL
FROM B2_OLAP.AR_ZKP_BY_EMPLOYEE@dwh
WHERE arcdate = :arcdate
  AND is_active_zkp = 'Y'
  AND is_employed = 1


import pandas as pd

dfs = []

for d in arcdates:
    df_zkp = pd.read_sql(
        QUERY,
        con=conn,
        params={"arcdate": d}
    )
    dfs.append(df_zkp)

zkp_df = pd.concat(dfs, ignore_index=True)
