sms_parts = []

for part in chunks(clients, 900):
    ids = sql_in(part)

    QUERY = f"""
        SELECT
            d.USERID AS CONTRAGENTID,
            d.ID AS DELIVERY_ID,
            d.IFOBSSMSTYPEID,
            d.PROCESSINGID,
            d.ADDRESS,
            d.CREATEDATE,
            d.SENDDATE,
            d.STATUS,
            d.TEXT AS DELIVERY_TEXT,
            mt.TEXT AS TEMPLATE_TEXT,
            mt.NAME AS SMS_TYPE_NAME
        FROM csbm.IF_IFOBSSMSDELIVERY@DWH d
        LEFT JOIN csbm.IF_MAILSMSTYPE@DWH mt
            ON d.IFOBSSMSTYPEID = mt.ID
        WHERE d.USERID IN ({ids})
    """

    sms_parts.append(get_data(QUERY))

sms_df = pd.concat(sms_parts, ignore_index=True) if sms_parts else pd.DataFrame()

sms_df.head()






push_parts = []

for part in chunks(clients, 900):
    ids = sql_in(part)

    QUERY = f"""
        SELECT
            p.USERID AS CONTRAGENTID,
            p.ID AS PUSH_ID,
            p.CREATEDATE,
            p.STATUS,
            p.TITLE,
            p.TEXT AS PUSH_TEXT
        FROM csbm.IF_PUSHMESSAGE@DWH p
        WHERE p.USERID IN ({ids})
    """

    push_parts.append(get_data(QUERY))

push_df = pd.concat(push_parts, ignore_index=True) if push_parts else pd.DataFrame()

push_df.head()


QUERY = """
SELECT *
FROM csbm.IF_PUSHMESSAGE@DWH
FETCH FIRST 5 ROWS ONLY
"""

get_data(QUERY).columns









if not sms_df.empty:
    sms_df["CONTRAGENTID"] = sms_df["CONTRAGENTID"].astype(str).str.strip()
    
    sms_agg = (
        sms_df
        .groupby("CONTRAGENTID", as_index=False)
        .agg(
            sms_was_sent=("DELIVERY_ID", "count"),
            sms_texts=("TEMPLATE_TEXT", lambda x: " | ".join(x.dropna().astype(str).unique())),
            sms_delivery_texts=("DELIVERY_TEXT", lambda x: " | ".join(x.dropna().astype(str).unique())),
            sms_addresses=("ADDRESS", lambda x: " | ".join(x.dropna().astype(str).unique()))
        )
    )
    
    sms_agg["has_sms"] = sms_agg["sms_was_sent"] > 0
else:
    sms_agg = pd.DataFrame(columns=[
        "CONTRAGENTID", "sms_was_sent", "sms_texts", 
        "sms_delivery_texts", "sms_addresses", "has_sms"
    ])



if not push_df.empty:
    push_df["CONTRAGENTID"] = push_df["CONTRAGENTID"].astype(str).str.strip()
    
    push_agg = (
        push_df
        .groupby("CONTRAGENTID", as_index=False)
        .agg(
            push_was_sent=("PUSH_ID", "count"),
            push_texts=("PUSH_TEXT", lambda x: " | ".join(x.dropna().astype(str).unique()))
        )
    )
    
    push_agg["has_push"] = push_agg["push_was_sent"] > 0
else:
    push_agg = pd.DataFrame(columns=[
        "CONTRAGENTID", "push_was_sent", "push_texts", "has_push"
    ])




result = clients_df.copy()

result = result.merge(
    sms_agg,
    on="CONTRAGENTID",
    how="left"
)

result = result.merge(
    push_agg,
    on="CONTRAGENTID",
    how="left"
)

result["has_sms"] = result["has_sms"].fillna(False)
result["has_push"] = result["has_push"].fillna(False)

result["sms_was_sent"] = result["sms_was_sent"].fillna(0).astype(int)
result["push_was_sent"] = result["push_was_sent"].fillna(0).astype(int)

result["was_contacted"] = result["has_sms"] | result["has_push"]

result.head()


