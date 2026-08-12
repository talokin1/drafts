ids = dataset['CONTRAGENTID'].dropna().astype(int).unique().tolist()

QUERY = f"""
SELECT
    ID,
    NAME
FROM your_table_name
WHERE ID IN ({','.join(map(str, ids))})
"""

names = get_data(QUERY)

dataset = dataset.merge(
    names,
    left_on='CONTRAGENTID',
    right_on='ID',
    how='left'
).drop(columns='ID')