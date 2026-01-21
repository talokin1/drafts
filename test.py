cmp = (
    df_nov[['CONTRAGENTID','TOTAL_SCORE',
            'BUSINESS_SCORE','PORTFOLIO_SCORE','DAILY_SCORE']]
    .merge(
        df_mar[['CONTRAGENTID','TOTAL_SCORE',
                'BUSINESS_SCORE','PORTFOLIO_SCORE','DAILY_SCORE']],
        on='CONTRAGENTID',
        suffixes=('_NOV','_MAR')
    )
)

for col in ['TOTAL_SCORE','BUSINESS_SCORE','PORTFOLIO_SCORE','DAILY_SCORE']:
    cmp[f'DELTA_{col}'] = cmp[f'{col}_MAR'] - cmp[f'{col}_NOV']


def growth_driver(row):
    deltas = {
        'BUSINESS'  : row['DELTA_BUSINESS_SCORE'],
        'PORTFOLIO' : row['DELTA_PORTFOLIO_SCORE'],
        'DAILY'     : row['DELTA_DAILY_SCORE'],
    }
    return max(deltas, key=deltas.get)
cmp['MAIN_GROWTH_DRIVER'] = cmp.apply(growth_driver, axis=1)



daily_delta = (
    df_mar[['CONTRAGENTID'] + DAILY_COLS]
    .set_index('CONTRAGENTID')
    - df_nov[['CONTRAGENTID'] + DAILY_COLS]
      .set_index('CONTRAGENTID')
).reset_index()

def top_daily_drivers(row, n=3):
    s = row[DAILY_COLS]
    return list(s.sort_values(ascending=False).head(n).index)

daily_delta['TOP_DAILY_DRIVERS'] = daily_delta.apply(top_daily_drivers, axis=1)




final = (
    cmp.merge(
        daily_delta[['CONTRAGENTID','TOP_DAILY_DRIVERS']],
        on='CONTRAGENTID',
        how='left'
    )
    .sort_values('DELTA_TOTAL_SCORE', ascending=False)
)
