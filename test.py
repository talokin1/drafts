sql = f"""
SELECT
    arcdate,
    id,
    bankaID,
    bankbID
FROM b2_olap.ar_document@dwh
WHERE arcdate >= TO_DATE('{arcdate.strftime("%d.%m.%Y")}', 'DD.MM.YYYY')
  AND arcdate <  TO_DATE('{(arcdate + pd.Timedelta(days=1)).strftime("%d.%m.%Y")}', 'DD.MM.YYYY')
  AND id IN ({ids_str})
"""
