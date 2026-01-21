def add_score_blocks(df):
    df = df.copy()
    df['BUSINESS_SCORE']  = df[BUSINESS_COLS].sum(axis=1)
    df['PORTFOLIO_SCORE'] = df[PORTFOLIO_COLS].sum(axis=1)
    df['DAILY_SCORE']     = df[DAILY_BANKING].sum(axis=1)
    return df

df_nov = add_score_blocks(df_nov)
df_mar = add_score_blocks(df_mar)



cmp = (
    df_nov[
        ['CONTRAGENTID','TOTAL_SCORE',
         'BUSINESS_SCORE','PORTFOLIO_SCORE','DAILY_SCORE']
    ]
    .merge(
        df_mar[
            ['CONTRAGENTID','TOTAL_SCORE',
             'BUSINESS_SCORE','PORTFOLIO_SCORE','DAILY_SCORE']
        ],
        on='CONTRAGENTID',
        suffixes=('_NOV', '_MAR')
    )
)


for col in ['TOTAL_SCORE','BUSINESS_SCORE','PORTFOLIO_SCORE','DAILY_SCORE']:
    cmp[f'DELTA_{col}'] = cmp[f'{col}_MAR'] - cmp[f'{col}_NOV']
def growth_driver(row):
    deltas = {
        'BUSINESS':  row['DELTA_BUSINESS_SCORE'],
        'PORTFOLIO': row['DELTA_PORTFOLIO_SCORE'],
        'DAILY':     row['DELTA_DAILY_SCORE'],
    }
    return max(deltas, key=deltas.get)
cmp['MAIN_GROWTH_DRIVER'] = cmp.apply(growth_driver, axis=1)


def build_delta(df_nov, df_mar, cols, prefix):
    delta = (
        df_mar[['CONTRAGENTID'] + cols]
        .set_index('CONTRAGENTID')
        - df_nov[['CONTRAGENTID'] + cols]
          .set_index('CONTRAGENTID')
    )
    delta.columns = [f'{prefix}_{c}' for c in delta.columns]
    return delta.reset_index()

business_delta  = build_delta(df_nov, df_mar, BUSINESS_COLS,  'BUSINESS')
portfolio_delta = build_delta(df_nov, df_mar, PORTFOLIO_COLS, 'PORTFOLIO')
daily_delta     = build_delta(df_nov, df_mar, DAILY_BANKING,  'DAILY')

cmp = cmp.merge(business_delta,  on='CONTRAGENTID', how='left')
cmp = cmp.merge(portfolio_delta, on='CONTRAGENTID', how='left')
cmp = cmp.merge(daily_delta,     on='CONTRAGENTID', how='left')



def extract_top_features(row, cols, n=3):
    s = row[cols]
    s = s[s > 0]
    if s.empty:
        return []
    return list(s.sort_values(ascending=False).head(n).index)

BUSINESS_DELTA_COLS  = [c for c in cmp.columns if c.startswith('BUSINESS_') and c not in ['BUSINESS_SCORE']]
PORTFOLIO_DELTA_COLS = [c for c in cmp.columns if c.startswith('PORTFOLIO_') and c not in ['PORTFOLIO_SCORE']]
DAILY_DELTA_COLS     = [c for c in cmp.columns if c.startswith('DAILY_') and c not in ['DAILY_SCORE']]


cmp['TOP_BUSINESS_DRIVERS'] = cmp.apply(
    extract_top_features,
    axis=1,
    cols=BUSINESS_DELTA_COLS
)

cmp['TOP_PORTFOLIO_DRIVERS'] = cmp.apply(
    extract_top_features,
    axis=1,
    cols=PORTFOLIO_DELTA_COLS
)

cmp['TOP_DAILY_DRIVERS'] = cmp.apply(
    extract_top_features,
    axis=1,
    cols=DAILY_DELTA_COLS
)


def explain_growth(row):
    if row['MAIN_GROWTH_DRIVER'] == 'BUSINESS':
        return f"BUSINESS: {row['TOP_BUSINESS_DRIVERS']}"
    if row['MAIN_GROWTH_DRIVER'] == 'PORTFOLIO':
        return f"PORTFOLIO: {row['TOP_PORTFOLIO_DRIVERS']}"
    return f"DAILY: {row['TOP_DAILY_DRIVERS']}"

cmp['GROWTH_EXPLANATION'] = cmp.apply(explain_growth, axis=1)
