# =========================
# Model analysis Excel report (2 sheets)
# Sheet 1: Validation (CNUM/ID, True, Pred, Diffs, Income_Range + optional cat cols)
# Sheet 2: Industry insights (GROUP_CODE / DIVISION_CODE etc.)
# =========================

import numpy as np
import pandas as pd

# ---------- helpers ----------
def build_income_bins(
    start=1000,
    stop=100001,
    step=5000,
    tail_start=100000,
    include_zero_bucket=True,
):
    """
    Returns (bins, labels) for pd.cut.
    Bucketization like:
      0-1k, 1k-6k, 6k-11k, ..., 96k-100k, 100k+
    """
    main_grid = np.arange(start, stop, step)  # 1000..100000
    bins = []
    if include_zero_bucket:
        bins.append(-1)  # will represent "0-1k"
    bins += list(main_grid)
    bins += [tail_start, np.inf]

    # unique + sorted (safety)
    bins = sorted(set(bins))

    labels = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if high == np.inf:
            labels.append(f"{int(low/1000)}k+")
        elif low == -1:
            labels.append(f"0-{int(high/1000)}k")
        else:
            labels.append(f"{int(low/1000)}k-{int(high/1000)}k")

    return bins, labels


def make_validation_report_df(
    validation_results: pd.DataFrame,
    X_meta: pd.DataFrame,
    id_col_pred="IDENTIFYCODE",
    include_cat_cols=True,
    cat_cols=None,
    true_col="True_Value",
    pred_col="Predicted",
):
    """
    validation_results must contain:
      - id_col_pred (e.g. 'IDENTIFYCODE') OR have it in index (then we reset_index)
      - true_col, pred_col
    X_meta is a frame indexed by IDENTIFYCODE with columns (cat cols + optional others).
    """
    vr = validation_results.copy()

    # Ensure ID column exists
    if id_col_pred not in vr.columns:
        if vr.index.name == id_col_pred:
            vr = vr.reset_index()
        else:
            # maybe it exists but was renamed differently
            vr = vr.reset_index().rename(columns={"index": id_col_pred})

    # Choose which meta columns to bring
    meta_cols = []
    if include_cat_cols:
        if cat_cols is None:
            # auto-detect categoricals from X_meta
            meta_cols = X_meta.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            meta_cols = list(cat_cols)

    # Merge meta
    meta = X_meta.copy()
    if meta.index.name != id_col_pred:
        meta.index.name = id_col_pred

    use_meta = meta[meta_cols].copy() if meta_cols else None
    if use_meta is not None and len(meta_cols) > 0:
        report_df = vr.merge(use_meta, left_on=id_col_pred, right_index=True, how="left")
    else:
        report_df = vr

    # Compute diffs
    report_df["Diffs"] = (report_df[true_col] - report_df[pred_col]).abs()

    # Add income range based on predicted
    bins, labels = build_income_bins()
    report_df["Income_Range"] = pd.cut(report_df[pred_col], bins=bins, labels=labels, include_lowest=True)

    return report_df


def make_industry_insights_sheet(
    report_df: pd.DataFrame,
    industry_cols,
    true_col="True_Value",
    pred_col="Predicted",
    min_count=30,
    top_n=20,
):
    """
    Returns a single dataframe suitable for Sheet 2:
    one block per industry feature (Feature_Name), top industries by Avg_Pred (potential).
    Also adds a "gap" = Avg_Pred - Avg_True to see over/under estimation.
    """
    blocks = []
    for col in industry_cols:
        if col not in report_df.columns:
            continue

        g = (
            report_df.groupby(col, observed=False)
            .agg(
                Count=(pred_col, "size"),
                Avg_Pred=(pred_col, "mean"),
                Med_Pred=(pred_col, "median"),
                Avg_True=(true_col, "mean"),
                Med_True=(true_col, "median"),
                MAE=("Diffs", "mean"),
                P90_True=(true_col, lambda x: x.quantile(0.9)),
            )
            .reset_index()
            .rename(columns={col: "Industry_Value"})
        )

        g = g[g["Count"] >= min_count].copy()
        if g.empty:
            continue

        g["Gap_PredMinusTrue"] = g["Avg_Pred"] - g["Avg_True"]
        g["Feature_Name"] = col

        # Order by business usefulness: expected potential (Avg_Pred) and stability (Count)
        g = g.sort_values(["Avg_Pred", "Count"], ascending=[False, False]).head(top_n)

        # Column order
        g = g[
            [
                "Feature_Name",
                "Industry_Value",
                "Count",
                "Avg_Pred",
                "Med_Pred",
                "Avg_True",
                "Med_True",
                "P90_True",
                "MAE",
                "Gap_PredMinusTrue",
            ]
        ]
        blocks.append(g)

    if blocks:
        out = pd.concat(blocks, ignore_index=True)
    else:
        out = pd.DataFrame(
            columns=[
                "Feature_Name",
                "Industry_Value",
                "Count",
                "Avg_Pred",
                "Med_Pred",
                "Avg_True",
                "Med_True",
                "P90_True",
                "MAE",
                "Gap_PredMinusTrue",
            ]
        )
    return out


def autosize_worksheet(worksheet, df, start_row=0, start_col=0, extra=2, max_width=60):
    """Simple autosize for xlsxwriter."""
    for j, col in enumerate(df.columns):
        series = df[col].astype(str)
        width = max(series.map(len).max(), len(str(col))) + extra
        width = min(width, max_width)
        worksheet.set_column(start_col + j, start_col + j, width)


ID_COL = "IDENTIFYCODE"

if "X_val" in globals():
    if X_val.index.name != ID_COL:
        if ID_COL in X_val.columns:
            X_val = X_val.set_index(ID_COL)
        else:
            X_val.index.name = ID_COL

report_df = make_validation_report_df(
    validation_results=validation_results,
    X_meta=X_val,
    id_col_pred=ID_COL,
    include_cat_cols=True,
    cat_cols=cat_cols if "cat_cols" in globals() else None,
    true_col="True_Value",
    pred_col="Predicted",
)

# Optional: add extra summary table (binned error) to Sheet 1 (placed at right)
summary_table = (
    report_df.groupby("Income_Range", observed=False)
    .agg(
        Count=("Diffs", "count"),
        Mean_Error=("Diffs", "mean"),
        Median_Error=("Diffs", "median"),
        Avg_Predicted=("Predicted", "mean"),
        Avg_True=("True_Value", "mean"),
    )
    .reset_index()
)
summary_table = summary_table.round(2)

# 2) Build Sheet 2 dataframe (industry insights)
# You can extend this list with other industry-like categorical cols you have
industry_cols_default = [c for c in ["GROUP_CODE", "DIVISION_CODE", "SECTION_CODE", "OPF_CODE"] if c in report_df.columns]
industry_cols = industry_cols_default

industry_table = make_industry_insights_sheet(
    report_df=report_df,
    industry_cols=industry_cols,
    true_col="True_Value",
    pred_col="Predicted",
    min_count=30,
    top_n=25,
).round(2)

# 3) Write Excel with formatting
file_name = "Model_Industry_Insights_Report.xlsx"

with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
    workbook = writer.book

    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D7E4BC", "border": 1})
    num_fmt = workbook.add_format({"num_format": "#,##0.00"})
    int_fmt = workbook.add_format({"num_format": "0"})
    money_fmt = workbook.add_format({"num_format": "#,##0.00"})

    # --- Sheet 1 ---
    sheet1 = "Validation_Report"
    report_df.to_excel(writer, sheet_name=sheet1, startrow=1, index=False)
    summary_table.to_excel(writer, sheet_name=sheet1, startrow=1, startcol=report_df.shape[1] + 2, index=False)

    ws1 = writer.sheets[sheet1]
    ws1.write(0, 0, "Client-level Validation (sample/full)", header_fmt)
    ws1.write(0, report_df.shape[1] + 2, "Binned Error Summary", header_fmt)

    # headers format
    for j, col in enumerate(report_df.columns):
        ws1.write(1, j, col, header_fmt)
    for j, col in enumerate(summary_table.columns):
        ws1.write(1, report_df.shape[1] + 2 + j, col, header_fmt)

    # autosize
    autosize_worksheet(ws1, report_df, start_row=1, start_col=0)
    autosize_worksheet(ws1, summary_table, start_row=1, start_col=report_df.shape[1] + 2)

    # numeric formatting (best-effort)
    # find numeric cols in report_df
    for j, col in enumerate(report_df.columns):
        if col in ["True_Value", "Predicted", "Diffs"]:
            ws1.set_column(j, j, None, money_fmt)

    # summary numeric cols
    base = report_df.shape[1] + 2
    for j, col in enumerate(summary_table.columns):
        if col in ["Count"]:
            ws1.set_column(base + j, base + j, None, int_fmt)
        elif col != "Income_Range":
            ws1.set_column(base + j, base + j, None, money_fmt)

    ws1.freeze_panes(2, 0)

    # --- Sheet 2 ---
    sheet2 = "Industry_Insights"
    industry_table.to_excel(writer, sheet_name=sheet2, startrow=1, index=False)
    ws2 = writer.sheets[sheet2]
    ws2.write(0, 0, "Top industries by predicted potential (and quality)", header_fmt)

    for j, col in enumerate(industry_table.columns):
        ws2.write(1, j, col, header_fmt)

    autosize_worksheet(ws2, industry_table, start_row=1, start_col=0)

    # format numeric columns
    for j, col in enumerate(industry_table.columns):
        if col == "Count":
            ws2.set_column(j, j, None, int_fmt)
        elif col in ["Feature_Name", "Industry_Value"]:
            pass
        else:
            ws2.set_column(j, j, None, money_fmt)

    ws2.freeze_panes(2, 0)

print(f"Report saved: {file_name}")
print("Sheet1 rows:", report_df.shape, "Sheet2 rows:", industry_table.shape)
