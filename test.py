unknown_target_mask = clients[target_source_cols].isna().any(axis=1)

clients['PACKAGE_FLAG'] = clients['PACKAGE'].eq(1).astype('int8')
clients['TOTAL_PORTFOLIO_FLAG'] = clients['TOTAL_PORTFOLIO'].gt(4_000_000).astype('int8')
clients['AUM_UAH_FLAG'] = clients['LIABILITIES_UAH'].gt(1_000_000).astype('int8')
clients['INCOME_FLAG'] = clients['INCOME(COM+INTEREST)'].gt(15_000).astype('int8')
clients['POS_FLAG'] = clients['AMT_DEB_CARD'].gt(50_000).astype('int8')

other_golden_flags = ['TOTAL_PORTFOLIO_FLAG', 'AUM_UAH_FLAG', 'INCOME_FLAG', 'POS_FLAG']

clients['OTHER_GOLDEN_COUNT'] = clients[other_golden_flags].sum(axis=1)
clients['GOLDEN_CRITERIA_COUNT'] = clients['PACKAGE_FLAG'] + clients['OTHER_GOLDEN_COUNT']
clients['GOLDEN_TARGET'] = (clients['PACKAGE_FLAG'].eq(1) & clients['OTHER_GOLDEN_COUNT'].ge(2)).astype('Int8')
clients.loc[unknown_target_mask, 'GOLDEN_TARGET'] = pd.NA

clients[target_source_cols] = clients[target_source_cols].fillna(0)

clients['GOLDEN_TARGET'].value_counts(dropna=False)
clients['GOLDEN_TARGET'].value_counts(normalize=True, dropna=False).mul(100).round(2)

pd.crosstab([clients['PACKAGE_FLAG'], clients['OTHER_GOLDEN_COUNT']], clients['GOLDEN_TARGET'], margins=True)