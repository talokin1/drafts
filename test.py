summary_cols = [
    'CONTRAGENTID',

    'TOTAL_SCORE_NOV',
    'TOTAL_SCORE_MAR',
    'DELTA_TOTAL_SCORE',

    'DELTA_BUSINESS_SCORE',
    'DELTA_PORTFOLIO_SCORE',
    'DELTA_DAILY_SCORE',

    'MAIN_GROWTH_DRIVER',
    'GROWTH_EXPLANATION'
]

summary = (
    cmp[summary_cols]
    .sort_values('DELTA_TOTAL_SCORE', ascending=False)
    .reset_index(drop=True)
)


def short_reason(row):
    if row['MAIN_GROWTH_DRIVER'] == 'BUSINESS':
        return f"Business → {row['DELTA_BUSINESS_SCORE']:+.0f}"
    if row['MAIN_GROWTH_DRIVER'] == 'PORTFOLIO':
        return f"Portfolio → {row['DELTA_PORTFOLIO_SCORE']:+.0f}"
    return f"Daily banking → {row['DELTA_DAILY_SCORE']:+.0f}"
summary['GROWTH_REASON_SHORT'] = summary.apply(short_reason, axis=1)

summary['TOTAL_SCORE_CHANGE'] = (
    summary['TOTAL_SCORE_NOV'].astype(int).astype(str)
    + ' → '
    + summary['TOTAL_SCORE_MAR'].astype(int).astype(str)
)

final_summary = summary[[
    'CONTRAGENTID',
    'TOTAL_SCORE_CHANGE',
    'DELTA_TOTAL_SCORE',
    'GROWTH_REASON_SHORT',
    'GROWTH_EXPLANATION'
]]
