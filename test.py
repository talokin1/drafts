cluster_growth = (
    cmp
    .groupby('CLUSTER')
    .agg(
        n_clients=('CONTRAGENTID', 'count'),
        total_growth=('DELTA_TOTAL_SCORE', 'sum'),
        avg_growth=('DELTA_TOTAL_SCORE', 'mean'),
        median_growth=('DELTA_TOTAL_SCORE', 'median')
    )
    .sort_values('total_growth', ascending=False)
)

cluster_block_driver = (
    cmp
    .groupby(['CLUSTER', 'GROWTH_BLOCK'])
    .size()
    .reset_index(name='cnt')
)
cluster_block_share = (
    cluster_block_driver
    .assign(
        share=lambda x: x['cnt'] /
        x.groupby('CLUSTER')['cnt'].transform('sum')
    )
    .sort_values(['CLUSTER','share'], ascending=[True, False])
)


factor_exploded = (
    cmp
    .explode('GROWTH_FACTORS')
    .dropna(subset=['GROWTH_FACTORS'])
)

cluster_factor_top = (
    factor_exploded
    .groupby(['CLUSTER', 'GROWTH_FACTORS'])
    .size()
    .reset_index(name='cnt')
    .sort_values(['CLUSTER','cnt'], ascending=[True, False])
)

top_factors_per_cluster = (
    cluster_factor_top
    .groupby('CLUSTER')
)



def build_cluster_insight(df):
    return (
        df
        .groupby('CLUSTER')
        .apply(lambda x:
            "; ".join(
                x['GROWTH_FACTORS'].value_counts().head(3).index
            )
        )
        .reset_index(name='KEY_GROWTH_DRIVERS')
    )
cluster_insights = build_cluster_insight(factor_exploded)



cluster_summary = (
    cluster_growth
    .reset_index()
    .merge(
        cluster_block_share
        .groupby('CLUSTER')
        .first()
        .reset_index()[['CLUSTER','GROWTH_BLOCK','share']],
        on='CLUSTER',
        how='left'
    )
    .merge(
        cluster_insights,
        on='CLUSTER',
        how='left'
    )
)
