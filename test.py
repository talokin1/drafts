import pandas as pd

output_file = "CPM_output_business.xlsx"

# порядок колонок
groups_dict = {
    "POTENTIAL": potential_cols,
    "COMPANY INFO": company_cols,
    "LIABILITIES MODEL": liabs_cols,
    "ASSETS MODEL": assets_cols,
    "FX MODEL": fx_cols,
}

# беремо тільки ті колонки, які реально є в df
ordered_cols = []
group_ranges = []

start_col = 0

for group_name, cols in groups_dict.items():
    existing_cols = [c for c in cols if c in df.columns]
    
    if len(existing_cols) == 0:
        continue
    
    end_col = start_col + len(existing_cols) - 1
    group_ranges.append((group_name, start_col, end_col))
    
    ordered_cols.extend(existing_cols)
    start_col = end_col + 1

result = df[ordered_cols].copy()

# бажано відсортувати для бізнесу
if "POTENTIAL_INCOME" in result.columns:
    result = result.sort_values("POTENTIAL_INCOME", ascending=False)

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    
    sheet_name = "Potential Clients"
    
    # пишемо дані без заголовків, бо заголовки зробимо вручну
    result.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
        header=False,
        startrow=2
    )
    
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    group_fmt = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "font_color": "white",
        "bg_color": "#1F4E78",
        "border": 1
    })
    
    header_fmt = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True,
        "bg_color": "#D9EAF7",
        "border": 1
    })
    
    money_fmt = workbook.add_format({
        "num_format": "#,##0.00",
        "border": 1
    })
    
    prob_fmt = workbook.add_format({
        "num_format": "0.00%",
        "border": 1
    })
    
    text_fmt = workbook.add_format({
        "border": 1
    })
    
    # 1-й рядок: групи
    for group_name, first_col, last_col in group_ranges:
        if first_col == last_col:
            worksheet.write(0, first_col, group_name, group_fmt)
        else:
            worksheet.merge_range(0, first_col, 0, last_col, group_name, group_fmt)
    
    # 2-й рядок: назви колонок
    for col_num, col_name in enumerate(result.columns):
        worksheet.write(1, col_num, col_name, header_fmt)
    
    # ширина колонок
    for col_num, col_name in enumerate(result.columns):
        if col_name in ["FIRM_NAME", "KVED_DESCR", "OPF_NAME", "DIVISION_NAME"]:
            worksheet.set_column(col_num, col_num, 32, text_fmt)
        elif col_name in ["IDENTIFYCODE", "CONTRAGENTID"]:
            worksheet.set_column(col_num, col_num, 16, text_fmt)
        elif "PROB" in col_name or col_name.endswith("_%"):
            worksheet.set_column(col_num, col_num, 14, prob_fmt)
        elif "POTENTIAL" in col_name or "AMT" in col_name or "INCOME" in col_name or "REVENUE" in col_name or "PROFIT" in col_name:
            worksheet.set_column(col_num, col_num, 18, money_fmt)
        else:
            worksheet.set_column(col_num, col_num, 16, text_fmt)
    
    # фільтри по другому рядку
    worksheet.autofilter(1, 0, len(result) + 1, len(result.columns) - 1)
    
    # закріпити групи + заголовки
    worksheet.freeze_panes(2, 0)
    
    # висота заголовків
    worksheet.set_row(0, 24)
    worksheet.set_row(1, 36)
    
    # умовне форматування для головного потенціалу
    if "POTENTIAL_INCOME" in result.columns:
        col_idx = result.columns.get_loc("POTENTIAL_INCOME")
        worksheet.conditional_format(
            2, col_idx,
            len(result) + 1, col_idx,
            {
                "type": "3_color_scale",
                "min_color": "#F8696B",
                "mid_color": "#FFEB84",
                "max_color": "#63BE7B"
            }
        )

print(f"Файл створено: {output_file}")