import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================
ID_COL = "IDENTIFYCODE"
TRUE_COL = "True_Value"
PRED_COL = "Predicted"
INDUSTRY_COL = "GROUP_CODE"  # можна змінити на DIVISION_CODE
OUTPUT_FILE = "Business_Model_Report.xlsx"

# =============================
# 1️⃣ BUILD REPORT DF
# =============================

# Ensure ID is column
vr = validation_results.copy()

if ID_COL not in vr.columns:
    vr = vr.reset_index().rename(columns={"index": ID_COL})

# Ensure X_val indexed by ID
if X_val.index.name != ID_COL:
    if ID_COL in X_val.columns:
        X_val = X_val.set_index(ID_COL)
    else:
        X_val.index.name = ID_COL

# Merge industry meta
meta_cols = X_val.select_dtypes(include=["object", "category"]).columns.tolist()
report_df = vr.merge(
    X_val[meta_cols],
    left_on=ID_COL,
    right_index=True,
    how="left"
)

# Absolute error
report_df["Abs_Error"] = (report_df[TRUE_COL] - report_df[PRED_COL]).abs()

# =============================
# 2️⃣ INCOME BUCKETS (BUSINESS STYLE)
# =============================

bins = [-1, 1000, 10000, 100000, 1000000, np.inf]
labels = ["0-1k", "1k-10k", "10k-100k", "100k-1M", "1M+"]

report_df["Income_Range"] = pd.cut(
    report_df[PRED_COL],
    bins=bins,
    labels=labels
)

# =============================
# 3️⃣ SHEET 1 SUMMARY TABLE
# =============================

summary_table = (
    report_df.groupby("Income_Range", observed=False)
    .agg(
        Clients=(ID_COL, "count"),
        Avg_Income=(TRUE_COL, "mean"),
        Avg_Predicted=(PRED_COL, "mean"),
        Avg_Error=("Abs_Error", "mean"),
        Median_Error=("Abs_Error", "median"),
    )
    .reset_index()
)

summary_table["Error_%"] = (
    summary_table["Avg_Error"] / summary_table["Avg_Income"]
) * 100

summary_table = summary_table.round(2)

# =============================
# 4️⃣ BUSINESS INDUSTRY REPORT
# =============================

df = report_df.copy()

# Top 20% threshold
high_threshold = df[TRUE_COL].quantile(0.8)
df["High_Value"] = df[TRUE_COL] >= high_threshold

industry_table = (
    df.groupby(INDUSTRY_COL, observed=False)
    .agg(
        Clients=(ID_COL, "count"),
        Avg_Income=(TRUE_COL, "mean"),
        Median_Income=(TRUE_COL, "median"),
        P90_Income=(TRUE_COL, lambda x: x.quantile(0.9)),
        Income_Std=(TRUE_COL, "std"),
        High_Value_Share=("High_Value", "mean"),
        Avg_Predicted=(PRED_COL, "mean"),
        Avg_Error=("Abs_Error", "mean")
    )
    .reset_index()
)

industry_table = industry_table[industry_table["Clients"] >= 30].copy()

industry_table["High_Value_%"] = industry_table["High_Value_Share"] * 100
industry_table["Prediction_Gap"] = (
    industry_table["Avg_Predicted"] - industry_table["Avg_Income"]
)

industry_table = industry_table.sort_values(
    "Avg_Income", ascending=False
).round(2)

# =============================
# 5️⃣ WRITE EXCEL
# =============================

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:

    workbook = writer.book

    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#D7E4BC",
        "border": 1
    })

    money_fmt = workbook.add_format({"num_format": "#,##0.00"})
    percent_fmt = workbook.add_format({"num_format": "0.00"})

    # --------------------------
    # SHEET 1: Validation
    # --------------------------

    sheet1 = "Validation_Report"

    report_df.to_excel(
        writer,
        sheet_name=sheet1,
        startrow=1,
        index=False
    )

    summary_table.to_excel(
        writer,
        sheet_name=sheet1,
        startrow=1,
        startcol=report_df.shape[1] + 2,
        index=False
    )

    ws1 = writer.sheets[sheet1]

    ws1.write(0, 0, "Client-level Prediction Report", header_fmt)
    ws1.write(0, report_df.shape[1] + 2, "Income Bucket Summary", header_fmt)

    # Format numeric columns
    for col_num, col_name in enumerate(report_df.columns):
        if col_name in [TRUE_COL, PRED_COL, "Abs_Error"]:
            ws1.set_column(col_num, col_num, 18, money_fmt)

    # --------------------------
    # SHEET 2: Industry Insights
    # --------------------------

    sheet2 = "Industry_Insights"

    industry_table.to_excel(
        writer,
        sheet_name=sheet2,
        startrow=1,
        index=False
    )

    ws2 = writer.sheets[sheet2]

    ws2.write(0, 0, "Industry Profitability & Concentration Report", header_fmt)

    for col_num, col_name in enumerate(industry_table.columns):

        if "Income" in col_name or "Predicted" in col_name or "Error" in col_name:
            ws2.set_column(col_num, col_num, 18, money_fmt)

        if "High_Value_%" in col_name:
            ws2.set_column(col_num, col_num, 15, percent_fmt)

    ws2.freeze_panes(2, 0)

print(f"Report saved to {OUTPUT_FILE}")
