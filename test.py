def top_drivers(df_nov, df_mar, cols, prefix, n=3):
    delta = (
        df_mar[['CONTRAGENTID'] + cols]
        .set_index('CONTRAGENTID')
        - df_nov[['CONTRAGENTID'] + cols]
          .set_index('CONTRAGENTID')
    )
    delta.columns = [f"{prefix}_{c}" for c in delta.columns]
    return delta.reset_index()

business_delta = top_drivers(
    df_nov, df_mar,
    BUSINESS_COLS,
    prefix='BUSINESS'
)

portfolio_delta = top_drivers(
    df_nov, df_mar,
    PORTFOLIO_COLS,
    prefix='PORTFOLIO'
)

daily_delta = top_drivers(
    df_nov, df_mar,
    DAILY_COLS,
    prefix='DAILY'
)


def extract_top_features(row, cols, n=3):
    s = row[cols]
    s = s[s > 0]           # тільки ті, що реально дали ріст
    return list(s.sort_values(ascending=False).head(n).index)


cmp = cmp.merge(business_delta, on='CONTRAGENTID')
cmp = cmp.merge(portfolio_delta, on='CONTRAGENTID')
cmp = cmp.merge(daily_delta, on='CONTRAGENTID')

cmp['TOP_BUSINESS_DRIVERS'] = cmp.apply(
    extract_top_features,
    axis=1,
    cols=[c for c in cmp.columns if c.startswith('BUSINESS_B')],
)

cmp['TOP_PORTFOLIO_DRIVERS'] = cmp.apply(
    extract_top_features,
    axis=1,
    cols=[c for c in cmp.columns if c.startswith('PORTFOLIO_P')],
)

cmp['TOP_DAILY_DRIVERS'] = cmp.apply(
    extract_top_features,
    axis=1,
    cols=[c for c in cmp.columns if c.startswith('DAILY_D')],
)


def explain_growth(row):
    drivers = {
        'BUSINESS': row['DELTA_BUSINESS_SCORE'],
        'PORTFOLIO': row['DELTA_PORTFOLIO_SCORE'],
        'DAILY': row['DELTA_DAILY_SCORE']
    }

    main_block = max(drivers, key=drivers.get)

    if main_block == 'BUSINESS':
        return f"BUSINESS: {row['TOP_BUSINESS_DRIVERS']}"
    if main_block == 'PORTFOLIO':
        return f"PORTFOLIO: {row['TOP_PORTFOLIO_DRIVERS']}"
    return f"DAILY: {row['TOP_DAILY_DRIVERS']}"

cmp['GROWTH_EXPLANATION'] = cmp.apply(explain_growth, axis=1)
