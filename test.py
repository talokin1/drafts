import pandas as pd
import numpy as np

output_path = r"M:\Controlling\Data_Science_Projects\Corp_Recommendation_Model\corp_recommendations.xlsx"

df = final.copy()

# порядок колонок
preferred_cols = [
    "IDENTIFYCODE",
    "FIRM_NAME",
    "KVED",
    "OPF_NAME",
    "FIRM_TYPE",
    "score_liabs",
    "score_assets",
    "score_fx",
    "recommended_product",
]
df = df[[c for c in preferred_cols if c in df.columns] + 
        [c for c in df.columns if c not in preferred_cols]]

# summary для бізнесу
total_clients = len(df)

product_summary = (
    df["recommended_product"]
    .value_counts(dropna=False)
    .rename_axis("Recommended product")
    .reset_index(name="Clients")
)

product_summary["Share"] = product_summary["Clients"] / total_clients

avg_scores = pd.DataFrame({
    "Metric": [
        "Total clients",
        "Avg Liabilities score",
        "Avg Assets score",
        "Avg FX score",
        "Max Liabilities score",
        "Max Assets score",
        "Max FX score",
    ],
    "Value": [
        total_clients,
        df["score_liabs"].mean(),
        df["score_assets"].mean(),
        df["score_fx"].mean(),
        df["score_liabs"].max(),
        df["score_assets"].max(),
        df["score_fx"].max(),
    ]
})

firm_type_summary = (
    df.groupby(["recommended_product", "FIRM_TYPE"], dropna=False)
    .size()
    .reset_index(name="Clients")
    .sort_values(["recommended_product", "Clients"], ascending=[True, False])
)

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Raw Data", index=False)
    
    workbook = writer.book
    ws_raw = writer.sheets["Raw Data"]

    # formats
    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#C6E0B4",   # світлозелений
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True
    })

    text_fmt = workbook.add_format({
        "border": 1,
        "valign": "top",
        "text_wrap": True
    })

    num_fmt = workbook.add_format({
        "border": 1,
        "num_format": "0.000",
        "valign": "top"
    })

    percent_fmt = workbook.add_format({
        "border": 1,
        "num_format": "0.0%",
        "valign": "top"
    })

    title_fmt = workbook.add_format({
        "bold": True,
        "font_size": 14,
        "bg_color": "#E2F0D9",
        "border": 1,
        "align": "center"
    })

    section_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#C6E0B4",
        "border": 1,
        "align": "center"
    })

    # Raw Data formatting
    for col_num, col_name in enumerate(df.columns):
        ws_raw.write(0, col_num, col_name, header_fmt)

        if col_name in ["score_liabs", "score_assets", "score_fx"]:
            ws_raw.set_column(col_num, col_num, 16, num_fmt)
        elif col_name in ["FIRM_NAME", "OPF_NAME"]:
            ws_raw.set_column(col_num, col_num, 35, text_fmt)
        elif col_name == "IDENTIFYCODE":
            ws_raw.set_column(col_num, col_num, 14, text_fmt)
        else:
            ws_raw.set_column(col_num, col_num, 18, text_fmt)

    ws_raw.freeze_panes(1, 0)
    ws_raw.autofilter(0, 0, len(df), len(df.columns) - 1)

    # Info sheet
    ws_info = workbook.add_worksheet("Info")

    ws_info.merge_range("A1:D1", "Corp Recommendation Engine — Business Summary", title_fmt)

    ws_info.write("A3", "Model logic", section_fmt)
    ws_info.write("A4", "Recommendation is based on three propensity models: Liabilities, Assets and FX.")
    ws_info.write("A5", "Each model score was normalized by its optimal threshold to make scores comparable.")
    ws_info.write("A6", "Final recommendation is the product with the highest normalized propensity score.")
    ws_info.write("A7", "Financial indicators are currently kept as explanation/context layer, not as weighted final-score component.")

    # KPI block
    ws_info.write("A10", "Key metrics", section_fmt)
    for r, row in enumerate(avg_scores.itertuples(index=False), start=11):
        ws_info.write(r, 0, row.Metric)
        ws_info.write(r, 1, row.Value, num_fmt if isinstance(row.Value, float) else text_fmt)

    # Product distribution
    start_row = 20
    ws_info.write(start_row, 0, "Recommendation distribution", section_fmt)
    product_summary.to_excel(
        writer,
        sheet_name="Info",
        startrow=start_row + 1,
        startcol=0,
        index=False
    )

    for col_num, col_name in enumerate(product_summary.columns):
        ws_info.write(start_row + 1, col_num, col_name, header_fmt)

    ws_info.set_column("A:A", 28)
    ws_info.set_column("B:B", 14)
    ws_info.set_column("C:C", 14, percent_fmt)

    # Firm type summary
    firm_start = start_row + len(product_summary) + 5
    ws_info.write(firm_start, 0, "Recommendation by firm type", section_fmt)

    firm_type_summary.to_excel(
        writer,
        sheet_name="Info",
        startrow=firm_start + 1,
        startcol=0,
        index=False
    )

    for col_num, col_name in enumerate(firm_type_summary.columns):
        ws_info.write(firm_start + 1, col_num, col_name, header_fmt)

    ws_info.set_column("A:A", 28)
    ws_info.set_column("B:B", 24)
    ws_info.set_column("C:C", 14)

    # Add chart for product distribution
    chart = workbook.add_chart({"type": "column"})
    chart.add_series({
        "name": "Clients",
        "categories": ["Info", start_row + 2, 0, start_row + 1 + len(product_summary), 0],
        "values":     ["Info", start_row + 2, 1, start_row + 1 + len(product_summary), 1],
    })
    chart.set_title({"name": "Clients by recommended product"})
    chart.set_x_axis({"name": "Product"})
    chart.set_y_axis({"name": "Clients"})
    chart.set_legend({"none": True})

    ws_info.insert_chart("E20", chart, {"x_scale": 1.25, "y_scale": 1.15})

    ws_info.freeze_panes(1, 0)

print(f"Excel saved to: {output_path}")