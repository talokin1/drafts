cmp['GROWTH_BLOCK'] = cmp['GROWTH_EXPLANATION_HUMAN'].str.split(':').str[0]

cmp['GROWTH_FACTORS_RAW'] = (
    cmp['GROWTH_EXPLANATION_HUMAN']
    .str.split(':', n=1)
    .str[1]
    .str.strip()
)



import ast

cmp['GROWTH_FACTORS'] = cmp['GROWTH_FACTORS_RAW'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)
cmp['N_GROWTH_FACTORS'] = cmp['GROWTH_FACTORS'].apply(len)
cmp['MAIN_GROWTH_FACTOR'] = cmp['GROWTH_FACTORS'].apply(
    lambda x: x[0] if x else None
)
cmp['SECOND_GROWTH_FACTOR'] = cmp['GROWTH_FACTORS'].apply(
    lambda x: x[1] if len(x) > 1 else None
)
cmp.explode('GROWTH_FACTORS')['GROWTH_FACTORS'].value_counts().head(10)

cmp['GROWTH_BLOCK'].value_counts(normalize=True)

cmp.explode('GROWTH_FACTORS') \
   .groupby(['CLUSTER','GROWTH_FACTORS']) \
   .size() \
   .sort_values(ascending=False)
