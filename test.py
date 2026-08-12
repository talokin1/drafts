import pandas as pd

ids = (
    dataset['CONTRAGENTID']
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

batch_size = 900
names_list = []

for i in range(0, len(ids), batch_size):
    batch = ids[i:i + batch_size]

    QUERY = f"""
    SELECT
        ID,
        NAME
    FROM your_table_name
    WHERE ID IN ({','.join(map(str, batch))})
    """

    names_list.append(get_data(QUERY))

names = pd.concat(names_list, ignore_index=True)

dataset = dataset.merge(
    names[['ID', 'NAME']],
    left_on='CONTRAGENTID',
    right_on='ID',
    how='left'
).drop(columns='ID')