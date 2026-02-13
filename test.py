import pandas as pd
import numpy as np

# =====================================
# CONFIG
# =====================================

ID_COL = "IDENTIFYCODE"
TRUE_COL = "True_Value"
PRED_COL = "Predicted"

INDUSTRY_COL = "GROUP_CODE"
OPF_COL = "OPF_CODE"

OUTPUT_FILE = "Business_Model_Report.xlsx"

# =====================================
# 1️⃣ PREPARE MAIN DF
# =====================================

df = validation_results.copy()

if ID_COL not in df.columns:
    df = df.reset_index().rename(columns={"index": ID_COL})

df["Abs_Error"] = (df[TRUE_COL] - df[PRED_COL]).abs()

# =====================================
# 2️⃣ BUCKETS
# =====================================

bins = [-1, 1000, 10000, 100000, 1000000, np.inf]
labels = ["0-1k", "1k-10k", "10k-100k", "100k-1M", "1M+"]

df["Income_Bucket_TRUE"] = pd.cut(df[TRUE_COL], bins=bins, labels=labels)
df["Income_Bucket_PRED"] = pd.cut(df[PRED_COL], bins=bins, labels=labels)

# =====================================
# 3️⃣ BUCKET TABLES
# =====================================

def bucket_summary(data, bucket_col, value_col):
    table = (
        data.groupby(bucket_col, observed=False)
        .agg(
            Clients=(ID_COL, "count"),
            Avg_Income=(value_col, "mean"),
            Median_Income=(value_col, "median"),
        )
        .reset_index()
    )
    return table.round(2)

bucket_true = bucket_summary(df, "Income_Bucket_TRUE", TRUE_COL)
bucket_pred = bucket_summary(df, "Income_Bucket_PRED", PRED_COL)

# =====================================
# 4️⃣ INDUSTRY INSIGHTS
# =====================================

def industry_report(data, group_col):

    high_threshold = data[TRUE_COL].quantile(0.8)
    data["High_Value"] = data[TRUE_COL] >= high_threshold

    g = (
        data.groupby(group_col, observed=False)
        .agg(
            Clients=(ID_COL, "count"),
            Avg_Income=(TRUE_COL, "mean"),
            Median_Income=(TRUE_COL, "median"),
            P90_Income=(TRUE_COL, lambda x: x.quantile(0.9)),
            High_Value_%=("High_Value", "mean"),
            Avg_Predicted=(PRED_COL, "mean"),
            Avg_Error=("Abs_Error", "mean")
        )
        .reset_index()
    )

    g["High_Value_%"] = g["High_Value_%"] * 100
    g["Prediction_Gap"] = g["Avg_Predicted"] - g["Avg_Income"]

    g = g[g["Clients"] >= 30]

    return g.sort_values("Avg_Income", ascending=False).round(2)

industry_table = industry_report(df.copy(), INDUSTRY_COL)
opf_table = industry_report(df.copy(), OPF_COL)

# =====================================
# 5️⃣ WRITE EXCEL
# =====================================

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:

    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D7E4BC"})
    money_fmt = workbook.add_format({"num_format": "#,##0.00"})
    percent_fmt = workbook.add_format({"num_format": "0.00"})

    # ---------------------------------
    # SHEET 1
    # ---------------------------------

    sheet1 = "Validation_Report"

    main_cols = [
        ID_COL,
        TRUE_COL,
        PRED_COL,
        "Abs_Error",
        "Income_Bucket_TRUE",
        "Income_Bucket_PRED"
    ]

    df[main_cols].to_excel(writer, sheet_name=sheet1, startrow=1, index=False)

    bucket_true.to_excel(writer, sheet_name=sheet1, startrow=1, startcol=8, index=False)
    bucket_pred.to_excel(writer, sheet_name=sheet1, startrow=1, startcol=14, index=False)

    ws1 = writer.sheets[sheet1]

    ws1.write(0, 0, "Client-Level Prediction", header_fmt)
    ws1.write(0, 8, "Buckets by TRUE", header_fmt)
    ws1.write(0, 14, "Buckets by PREDICTED", header_fmt)

    for col_num, col_name in enumerate(main_cols):
        if col_name in [TRUE_COL, PRED_COL, "Abs_Error"]:
            ws1.set_column(col_num, col_num, 18, money_fmt)

    ws1.freeze_panes(2, 0)

    # ---------------------------------
    # SHEET 2
    # ---------------------------------

    sheet2 = "Industry_Insights"
    industry_table.to_excel(writer, sheet_name=sheet2, startrow=1, index=False)

    ws2 = writer.sheets[sheet2]
    ws2.write(0, 0, "Industry Profitability Report", header_fmt)

    for col_num, col_name in enumerate(industry_table.columns):

        if "Income" in col_name or "Predicted" in col_name or "Error" in col_name:
            ws2.set_column(col_num, col_num, 18, money_fmt)

        if "High_Value_%" in col_name:
            ws2.set_column(col_num, col_num, 15, percent_fmt)

    ws2.freeze_panes(2, 0)

    # ---------------------------------
    # SHEET 3
    # ---------------------------------

    sheet3 = "OPF_Insights"
    opf_table.to_excel(writer, sheet_name=sheet3, startrow=1, index=False)

    ws3 = writer.sheets[sheet3]
    ws3.write(0, 0, "OPF Profitability Report", header_fmt)

    for col_num, col_name in enumerate(opf_table.columns):

        if "Income" in col_name or "Predicted" in col_name or "Error" in col_name:
            ws3.set_column(col_num, col_num, 18, money_fmt)

        if "High_Value_%" in col_name:
            ws3.set_column(col_num, col_num, 15, percent_fmt)

    ws3.freeze_panes(2, 0)

print(f"Report saved to {OUTPUT_FILE}")
