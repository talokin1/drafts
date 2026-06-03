potential_cols = [
    'POTENTIAL_INCOME',
    'FX_POTENTIAL',
    'TRANSACTION_POTENTIAL',
    'ACCOUNTS_POTENTIAL',
    'ASSETS_POTENTIAL',
    'LIABILITIES_POTENTIAL'
]


company_cols = [
    'IDENTIFYCODE',
    'FIRM_NAME',
    'DIVISION_CODE',
    'DIVISION_NAME',
    'KVED',
    'KVED_DESCR',
    'OPF_CODE',
    'OPF_NAME',
    'FIRM_TYPE',
    'NB_EMPL',
    'CONTRAGENTID',
    'MONTHLY_INCOME',
    'REVENUE_CUR',
    'NET_PROFIT_CUR',
    'WC_MAX_AMT'
]



liabs_cols = [
    'PRIMARY_LIABS',
    '0_LIABS',
    '0-100K_LIABS',
    '100K-500K_LIABS',
    '500K-1M_LIABS',
    '1M-5M_LIABS',
    '5M-10M_LIABS',
    '10M+_LIABS'
]


assets_cols = [
    'PRIMARY_ASSETS',
    '0_ASSETS',
    '0-5M_ASSETS',
    '5M-10M_ASSETS',
    '10M-20M_ASSETS',
    '20M-30M_ASSETS',
    '>30M_ASSETS',
    'ASSETS_SUIT_AMT',
    'ASSETS_SUIT_AMT_%'
]

fx_cols = [
    'EXPORT_USD',
    'IMPORT_USD',
    'PROB_TO_FX'
]


import pandas as pd

groups = (
    [('POTENTIAL', c) for c in potential_cols] +
    [('COMPANY INFO', c) for c in company_cols] +
    [('LIABILITIES MODEL', c) for c in liabs_cols] +
    [('ASSETS MODEL', c) for c in assets_cols] +
    [('FX MODEL', c) for c in fx_cols]
)

ordered_cols = [c[1] for c in groups]

result = df[ordered_cols].copy()

result.columns = pd.MultiIndex.from_tuples(groups)

with pd.ExcelWriter(
    'Corp_Potential_Model.xlsx',
    engine='xlsxwriter'
) as writer:

    result.to_excel(
        writer,
        sheet_name='Potential Clients',
        index=False
    )

    workbook = writer.book
    worksheet = writer.sheets['Potential Clients']

    header_fmt = workbook.add_format({
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#D9EAD3',
        'border': 1
    })

    for col_num, value in enumerate(result.columns):
        worksheet.set_column(col_num + 1, col_num + 1, 18)

    worksheet.freeze_panes(2, 0)