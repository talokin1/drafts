def extract_articles(df, block, col, mapping):
    tmp = (
        df[df['GROWTH_BLOCK'] == block]
        .explode(col)
        .dropna(subset=[col])
    )

    return (
        tmp
        .groupby(['CLUSTER', col])
        .size()
        .reset_index(name='cnt')
        .assign(
            article=lambda x: x[col].map(mapping)
        )
        .sort_values(['CLUSTER','cnt'], ascending=[True, False])
    )


business_articles = extract_articles(
    cmp,
    block='Business',
    col='TOP_BUSINESS_DRIVERS',
    mapping=BUSINESS_MAP
)

portfolio_articles = extract_articles(
    cmp,
    block='Portfolio',
    col='TOP_PORTFOLIO_DRIVERS',
    mapping=PORTFOLIO_MAP
)

daily_articles = extract_articles(
    cmp,
    block='Daily banking',
    col='TOP_DAILY_DRIVERS',
    mapping=DAILY_MAP
)

business_insights = (
    business_articles
    .groupby('CLUSTER')
    .head(3)
    [['CLUSTER','article','cnt']]
)

portfolio_insights = (
    portfolio_articles
    .groupby('CLUSTER')
    .head(3)
    [['CLUSTER','article','cnt']]
)

daily_insights = (
    daily_articles
    .groupby('CLUSTER')
    .head(3)
    [['CLUSTER','article','cnt']]
)


cluster_articles_summary = {
    'Business': business_insights,
    'Portfolio': portfolio_insights,
    'Daily banking': daily_insights
}
