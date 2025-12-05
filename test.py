import pandas as pd

dfs = {
    'jan_liabs_variant_1': jan_liabs,
    'feb_liabs_variant_1': feb_liabs,
    'mar_liabs_variant_1': mar_liabs,
    'apr_liabs_variant_1': apr_liabs,
    'may_liabs_variant_1': may_liabs,
    'jun_liabs_variant_1': jun_liabs,
    'jul_liabs_variant_1': jul_liabs,
    'aug_liabs_variant_1': aug_liabs,

    'jan_assets_variant_1': jan_assets,
    'feb_assets_variant_1': feb_assets,
    'mar_assets_variant_1': mar_assets,
    'apr_assets_variant_1': apr_assets,
    'may_assets_variant_1': may_assets,
    'jun_assets_variant_1': jun_assets,
    'jul_assets_variant_1': jul_assets,
    'aug_assets_variant_1': aug_assets,
}

with pd.ExcelWriter('FTP_Variant_1.xlsx', engine='xlsxwriter') as writer:
    for sheet_name, df in dfs.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Автоширина колонок
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, max_len + 2)

        # Стилізована таблиця
        worksheet.add_table(
            0, 0,
            df.shape[0], df.shape[1]-1,
            {
                'columns': [{'header': col} for col in df.columns],
                'style': 'Table Style Medium 2'
            }
        )
