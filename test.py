import pandas as pd
import numpy as np

ID_COL = "IDENTIFYCODE"
TRUE_COL = "True_Value"
PRED_COL = "Predicted"

# 1. Використовуємо DIVISION_CODE для агрегації макро-індустрій
INDUSTRY_COL = "DIVISION_CODE"
OPF_COL = "OPF_CODE"

OUTPUT_FILE = "Business_Model_Report_v5_Final.xlsx"

df = validation_results.copy()

# Перевірка наявності ID
if ID_COL not in df.columns:
    df = df.reset_index().rename(columns={"index": ID_COL})

# Підтягуємо фічі з X_val
if ID_COL not in X_val.columns and X_val.index.name == ID_COL:
    df = df.merge(
        X_val[[INDUSTRY_COL, OPF_COL]],
        left_on=ID_COL,
        right_index=True,
        how="left"
    )
else:
    df = df.merge(
        X_val[[ID_COL, INDUSTRY_COL, OPF_COL]],
        on=ID_COL,
        how="left"
    )

df["Abs_Error"] = (df[TRUE_COL] - df[PRED_COL]).abs()

# 2. Нові бакети з акцентом на Micro (1.5k) та Corp (2.5k)
bins = [-1, 1000, 2500, 10000]
bins += list(range(20000, 100001, 10000))
bins += [1000000, np.inf]
bins = sorted(set(bins))

labels = []
for i in range(len(bins)-1):
    low = bins[i]
    high = bins[i+1]
    
    if high == np.inf:
        labels.append(f"{int(low/1000)}k+")
    elif low == -1:
        labels.append(f"0-{high/1000:g}k") # Використовуємо :g щоб відкинути нулі після крапки
    else:
        labels.append(f"{low/1000:g}k-{high/1000:g}k")

df["Income_Bucket_TRUE"] = pd.cut(df[TRUE_COL], bins=bins, labels=labels)
df["Income_Bucket_PRED"] = pd.cut(df[PRED_COL], bins=bins, labels=labels)

def bucket_summary(data, bucket_col, value_col):
    table = (
        data.groupby(bucket_col, observed=False)
        .agg(
            Clients=(ID_COL, "count"),
            Avg_Income=(value_col, "mean"),
            Median_Income=(value_col, "median")
        )
        .reset_index()
    )
    return table.round(2)

bucket_true = bucket_summary(df, "Income_Bucket_TRUE", TRUE_COL)
bucket_pred = bucket_summary(df, "Income_Bucket_PRED", PRED_COL)


# 3. Звіт по індустріям та ОПФ (БЕЗ втрати клієнтів)
def segment_report(data, group_col):
    high_threshold = data[TRUE_COL].quantile(0.8)
    data["High_Value"] = data[TRUE_COL] >= high_threshold
    
    g = (
        data.groupby(group_col, observed=False)
        .agg(
            Clients=(ID_COL, "count"),
            Avg_Income=(TRUE_COL, "mean"),
            Median_Income=(TRUE_COL, "median"),
            P90_Income=(TRUE_COL, lambda x: x.quantile(0.9)),
            Avg_Predicted=(PRED_COL, "mean"),
            Avg_Error=("Abs_Error", "mean"),
        )
        .reset_index()
    )
    
    g["Prediction_Gap"] = g["Avg_Predicted"] - g["Avg_Income"]
    
    # Фільтр g = g[g["Clients"] >= 10] видалено, щоб усі клієнти потрапили у звіт.
    
    return g.sort_values("Avg_Income", ascending=False).round(2)

industry_table = segment_report(df.copy(), INDUSTRY_COL)
opf_table = segment_report(df.copy(), OPF_COL)


# 4. Експорт у Excel
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D7E4BC"})
    money_fmt = workbook.add_format({"num_format": "#,##0.00"})
    percent_fmt = workbook.add_format({"num_format": "0.00%"})
    
    # --- Sheet 1: Validation Report ---
    sheet1 = "Validation_Report"
    main_cols = [
        ID_COL, TRUE_COL, PRED_COL, "Abs_Error", 
        "Income_Bucket_TRUE", "Income_Bucket_PRED"
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
    
    # --- Sheet 2: Industry Insights ---
    sheet2 = "Industry_Insights"
    industry_table.to_excel(writer, sheet_name=sheet2, startrow=1, index=False)
    
    ws2 = writer.sheets[sheet2]
    ws2.write(0, 0, "Industry Profitability Report", header_fmt)
    
    for col_num, col_name in enumerate(industry_table.columns):
        if any(keyword in col_name for keyword in ["Income", "Predicted", "Error", "Gap"]):
            ws2.set_column(col_num, col_num, 18, money_fmt)
        if "High_Value_%" in col_name:
            ws2.set_column(col_num, col_num, 15, percent_fmt)
    ws2.freeze_panes(2, 0)
    
    # --- Sheet 3: OPF Insights ---
    sheet3 = "OPF_Insights"
    opf_table.to_excel(writer, sheet_name=sheet3, startrow=1, index=False)
    
    ws3 = writer.sheets[sheet3]
    ws3.write(0, 0, "OPF Profitability Report", header_fmt)
    
    for col_num, col_name in enumerate(opf_table.columns):
        if any(keyword in col_name for keyword in ["Income", "Predicted", "Error", "Gap"]):
            ws3.set_column(col_num, col_num, 18, money_fmt)
        if "High_Value_%" in col_name:
            ws3.set_column(col_num, col_num, 15, percent_fmt)
    ws3.freeze_panes(2, 0)

print(f"Report saved to {OUTPUT_FILE}")