import numpy as np
import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import (
    Font, PatternFill, Border, Side, Alignment
)
from openpyxl.utils import get_column_letter


# ============================================================
# 1. Settings
# ============================================================

INPUT_FILE = "your_file.xlsx"   # <-- replace with your file name
INPUT_SHEET = "Model Accuracy"

OUTPUT_FILE = "peter_classification_report.xlsx"
REPORT_SHEET = "Peter Classification Report"

FACT_COL = "MONTHLY_INCOME"
PRED_COL = "POTENTIAL_INCOME"

bins = [-np.inf, 300, 1000, 2500, 10_000, np.inf]

labels = [
    "total hopeless",
    "bad quality",
    "grey zone",
    "green zone",
    "fantastic"
]

class_rank = {
    "total hopeless": 0,
    "bad quality": 1,
    "grey zone": 2,
    "green zone": 3,
    "fantastic": 4
}


# ============================================================
# 2. Helper functions
# ============================================================

def clean_numeric(series):
    return (
        series
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace("-", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )


def write_df_to_sheet(ws, df, start_row, start_col, title=None):
    """
    Writes DataFrame to worksheet with optional section title.
    Returns next available row.
    """
    row = start_row

    if title:
        ws.cell(row=row, column=start_col, value=title)
        ws.cell(row=row, column=start_col).font = Font(
            bold=True, size=14, color="FFFFFF"
        )
        ws.cell(row=row, column=start_col).fill = PatternFill(
            "solid", fgColor="1F4E79"
        )
        ws.cell(row=row, column=start_col).alignment = Alignment(
            horizontal="left", vertical="center"
        )

        end_col = start_col + len(df.columns) - 1
        ws.merge_cells(
            start_row=row,
            start_column=start_col,
            end_row=row,
            end_column=end_col
        )

        row += 1

    # Header
    for j, col_name in enumerate(df.columns, start=start_col):
        cell = ws.cell(row=row, column=j, value=col_name)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Data
    for i, (_, data_row) in enumerate(df.iterrows(), start=row + 1):
        for j, value in enumerate(data_row, start=start_col):
            cell = ws.cell(row=i, column=j, value=value)

            if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                cell.number_format = "#,##0.00" if isinstance(value, float) else "#,##0"

            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    return row + len(df) + 3


def apply_table_style(ws, min_row, max_row, min_col, max_col):
    thin_gray = Side(style="thin", color="D9E2F3")
    border = Border(
        left=thin_gray,
        right=thin_gray,
        top=thin_gray,
        bottom=thin_gray
    )

    for row in ws.iter_rows(
        min_row=min_row,
        max_row=max_row,
        min_col=min_col,
        max_col=max_col
    ):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )

            if cell.row > min_row:
                cell.fill = PatternFill("solid", fgColor="FFFFFF")


def color_confusion_matrix(ws, start_row, start_col, n_classes=5):
    """
    Colors 5x5 confusion matrix.
    Diagonal = green.
    Near diagonal = light green/yellow.
    Far errors = red/orange.
    """
    for i in range(n_classes):
        for j in range(n_classes):
            cell = ws.cell(
                row=start_row + 1 + i,
                column=start_col + 1 + j
            )

            distance = abs(i - j)

            if distance == 0:
                color = "A9D18E"  # green
            elif distance == 1:
                color = "FFF2CC"  # yellow
            elif distance == 2:
                color = "F8CBAD"  # orange
            else:
                color = "F4CCCC"  # red

            cell.fill = PatternFill("solid", fgColor=color)
            cell.font = Font(bold=(distance == 0))


# ============================================================
# 3. Read and prepare data
# ============================================================

df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
df = df.dropna(how="all").copy()

df[FACT_COL] = clean_numeric(df[FACT_COL])
df[PRED_COL] = clean_numeric(df[PRED_COL])

eval_df = df.dropna(subset=[FACT_COL, PRED_COL]).copy()

# Main validation logic: only clients with known non-zero income
eval_df = eval_df[eval_df[FACT_COL] != 0].copy()

eval_df["Actual Peter Class"] = pd.cut(
    eval_df[FACT_COL],
    bins=bins,
    labels=labels,
    right=False
)

eval_df["Predicted Peter Class"] = pd.cut(
    eval_df[PRED_COL],
    bins=bins,
    labels=labels,
    right=False
)

eval_df["Actual Class Rank"] = eval_df["Actual Peter Class"].map(class_rank).astype(int)
eval_df["Predicted Class Rank"] = eval_df["Predicted Peter Class"].map(class_rank).astype(int)

eval_df["Class Distance"] = (
    eval_df["Actual Class Rank"] - eval_df["Predicted Class Rank"]
).abs()


# ============================================================
# 4. Metrics
# ============================================================

total_count = len(eval_df)

correct_count = (
    eval_df["Actual Peter Class"] == eval_df["Predicted Peter Class"]
).sum()

overall_accuracy = correct_count / total_count

same_or_neighbor_count = (eval_df["Class Distance"] <= 1).sum()
same_or_neighbor_accuracy = same_or_neighbor_count / total_count

mae_non_zero = np.mean(
    np.abs(eval_df[FACT_COL] - eval_df[PRED_COL])
)


summary_metrics = pd.DataFrame({
    "Metric": [
        "Validation Clients Count",
        "Correct Peter Class Count",
        "Overall Peter Class Accuracy",
        "Same or Neighbor Class Count",
        "Same or Neighbor Class Accuracy",
        "MAE for Non-Zero Clients"
    ],
    "Value": [
        total_count,
        correct_count,
        overall_accuracy,
        same_or_neighbor_count,
        same_or_neighbor_accuracy,
        mae_non_zero
    ],
    "Slide Format": [
        f"{total_count:,.0f}".replace(",", " "),
        f"{correct_count:,.0f}".replace(",", " "),
        f"{overall_accuracy * 100:.1f}%",
        f"{same_or_neighbor_count:,.0f}".replace(",", " "),
        f"{same_or_neighbor_accuracy * 100:.1f}%",
        f"{mae_non_zero:,.0f}".replace(",", " ")
    ]
})


slide_kpis = pd.DataFrame({
    "KPI": [
        "Validation Base",
        "Exact Peter Class Match",
        "Same or Neighbor Class Match",
        "MAE for Non-Zero Clients"
    ],
    "Value": [
        f"{total_count:,.0f}".replace(",", " "),
        f"{overall_accuracy * 100:.1f}% ({correct_count:,.0f} clients)".replace(",", " "),
        f"{same_or_neighbor_accuracy * 100:.1f}% ({same_or_neighbor_count:,.0f} clients)".replace(",", " "),
        f"{mae_non_zero:,.0f}".replace(",", " ")
    ]
})


# ============================================================
# 5. Confusion matrices
# ============================================================

conf_matrix = pd.crosstab(
    eval_df["Actual Peter Class"],
    eval_df["Predicted Peter Class"],
    rownames=["Actual Class"],
    colnames=["Predicted Class"],
    dropna=False
)

conf_matrix = conf_matrix.reindex(index=labels, columns=labels, fill_value=0)

conf_matrix_report = conf_matrix.copy()
conf_matrix_report["Grand Total"] = conf_matrix_report.sum(axis=1)
conf_matrix_report.loc["Grand Total"] = conf_matrix_report.sum(axis=0)

conf_matrix_report = conf_matrix_report.reset_index()


conf_matrix_pct_total = conf_matrix / conf_matrix.values.sum()

conf_matrix_pct_report = conf_matrix_pct_total.copy()
conf_matrix_pct_report["Grand Total"] = conf_matrix_pct_report.sum(axis=1)
conf_matrix_pct_report.loc["Grand Total"] = conf_matrix_pct_report.sum(axis=0)

conf_matrix_pct_report = (conf_matrix_pct_report * 100).round(2)
conf_matrix_pct_report = conf_matrix_pct_report.reset_index()


# ============================================================
# 6. Accuracy by class
# ============================================================

class_metrics = []

for cls in labels:
    fact_count = conf_matrix.loc[cls].sum()
    correct_cls_count = conf_matrix.loc[cls, cls]

    class_accuracy = correct_cls_count / fact_count if fact_count > 0 else np.nan

    class_metrics.append({
        "Peter Class": cls,
        "Actual Clients Count": fact_count,
        "Correct Predictions Count": correct_cls_count,
        "Class Accuracy": class_accuracy,
        "Class Accuracy, %": round(class_accuracy * 100, 2)
    })

class_metrics_df = pd.DataFrame(class_metrics)


# ============================================================
# 7. Save raw calculations to temporary Excel
# ============================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    pd.DataFrame().to_excel(writer, sheet_name=REPORT_SHEET, index=False)


# ============================================================
# 8. Build one-page formatted report
# ============================================================

wb = load_workbook(OUTPUT_FILE)
ws = wb[REPORT_SHEET]

# Remove default empty first row
ws.delete_rows(1, 1)

# White background
for row in range(1, 90):
    for col in range(1, 16):
        ws.cell(row=row, column=col).fill = PatternFill("solid", fgColor="FFFFFF")

# Title
ws.merge_cells("A1:L1")
ws["A1"] = "Peter Classification Model Accuracy Report"
ws["A1"].font = Font(bold=True, size=20, color="1F1F1F")
ws["A1"].alignment = Alignment(horizontal="left", vertical="center")

ws.merge_cells("A2:L2")
ws["A2"] = "Validation of predicted potential income converted into Peter business classes"
ws["A2"].font = Font(size=11, color="666666")
ws["A2"].alignment = Alignment(horizontal="left", vertical="center")


# KPI cards
kpi_start_row = 4
kpi_start_col = 1

kpi_cards = [
    ("Validation Base", f"{total_count:,.0f}".replace(",", " ")),
    ("Exact Class Match", f"{overall_accuracy * 100:.1f}%"),
    ("Same / Neighbor Match", f"{same_or_neighbor_accuracy * 100:.1f}%"),
    ("MAE Non-Zero", f"{mae_non_zero:,.0f}".replace(",", " "))
]

card_width = 3

for idx, (name, value) in enumerate(kpi_cards):
    col = kpi_start_col + idx * card_width

    ws.merge_cells(
        start_row=kpi_start_row,
        start_column=col,
        end_row=kpi_start_row,
        end_column=col + card_width - 2
    )

    ws.cell(kpi_start_row, col).value = name
    ws.cell(kpi_start_row, col).font = Font(bold=True, size=10, color="FFFFFF")
    ws.cell(kpi_start_row, col).fill = PatternFill("solid", fgColor="1F4E79")
    ws.cell(kpi_start_row, col).alignment = Alignment(horizontal="center")

    ws.merge_cells(
        start_row=kpi_start_row + 1,
        start_column=col,
        end_row=kpi_start_row + 2,
        end_column=col + card_width - 2
    )

    ws.cell(kpi_start_row + 1, col).value = value
    ws.cell(kpi_start_row + 1, col).font = Font(bold=True, size=18, color="006100")
    ws.cell(kpi_start_row + 1, col).fill = PatternFill("solid", fgColor="E2F0D9")
    ws.cell(kpi_start_row + 1, col).alignment = Alignment(horizontal="center", vertical="center")


# Tables
row = 9

summary_start = row
row = write_df_to_sheet(
    ws,
    slide_kpis,
    start_row=row,
    start_col=1,
    title="Main Metrics for Slide"
)
apply_table_style(ws, summary_start + 1, row - 3, 1, 2)

class_start = row
row = write_df_to_sheet(
    ws,
    class_metrics_df,
    start_row=row,
    start_col=1,
    title="Accuracy by Actual Peter Class"
)
apply_table_style(ws, class_start + 1, row - 3, 1, len(class_metrics_df.columns))

conf_start = 9
write_df_to_sheet(
    ws,
    conf_matrix_report,
    start_row=conf_start,
    start_col=8,
    title="Confusion Matrix: Counts"
)

apply_table_style(
    ws,
    conf_start + 1,
    conf_start + 1 + len(conf_matrix_report),
    8,
    8 + len(conf_matrix_report.columns) - 1
)

color_confusion_matrix(
    ws,
    start_row=conf_start + 1,
    start_col=8,
    n_classes=5
)


pct_start = conf_start + 11
write_df_to_sheet(
    ws,
    conf_matrix_pct_report,
    start_row=pct_start,
    start_col=8,
    title="Confusion Matrix: % of Total Validation Base"
)

apply_table_style(
    ws,
    pct_start + 1,
    pct_start + 1 + len(conf_matrix_pct_report),
    8,
    8 + len(conf_matrix_pct_report.columns) - 1
)

color_confusion_matrix(
    ws,
    start_row=pct_start + 1,
    start_col=8,
    n_classes=5
)


summary_full_start = row
row = write_df_to_sheet(
    ws,
    summary_metrics,
    start_row=row,
    start_col=1,
    title="Full Summary Metrics"
)

apply_table_style(
    ws,
    summary_full_start + 1,
    row - 3,
    1,
    len(summary_metrics.columns)
)


# Insight box
insight_row = pct_start + 12

ws.merge_cells(
    start_row=insight_row,
    start_column=8,
    end_row=insight_row + 4,
    end_column=14
)

insight_text = (
    f"Business interpretation:\n"
    f"{same_or_neighbor_accuracy * 100:.1f}% of clients are classified into the actual "
    f"Peter class or a neighboring class. This means the model is suitable as a "
    f"business prioritization and ranking tool, not as an exact accounting-level "
    f"income forecast."
)

ws.cell(insight_row, 8).value = insight_text
ws.cell(insight_row, 8).font = Font(size=11, color="1F1F1F", bold=False)
ws.cell(insight_row, 8).fill = PatternFill("solid", fgColor="E2F0D9")
ws.cell(insight_row, 8).alignment = Alignment(
    horizontal="left",
    vertical="center",
    wrap_text=True
)

thin_green = Side(style="thin", color="70AD47")
for r in range(insight_row, insight_row + 5):
    for c in range(8, 15):
        ws.cell(r, c).border = Border(
            left=thin_green,
            right=thin_green,
            top=thin_green,
            bottom=thin_green
        )


# Number formats
for row_cells in ws.iter_rows():
    for cell in row_cells:
        if isinstance(cell.value, float):
            if "Accuracy" in str(ws.cell(row=cell.row, column=1).value):
                cell.number_format = "0.00%"
            else:
                cell.number_format = "#,##0.00"


# Better formatting for accuracy columns
for row_cells in ws.iter_rows():
    for cell in row_cells:
        if cell.value in ["Class Accuracy"]:
            col_idx = cell.column
            for r in range(cell.row + 1, cell.row + 10):
                ws.cell(r, col_idx).number_format = "0.00%"


# Column widths
widths = {
    "A": 28,
    "B": 28,
    "C": 22,
    "D": 22,
    "E": 20,
    "F": 18,
    "G": 4,
    "H": 24,
    "I": 17,
    "J": 17,
    "K": 17,
    "L": 17,
    "M": 17,
    "N": 17
}

for col, width in widths.items():
    ws.column_dimensions[col].width = width

# Row heights
for r in range(1, 90):
    ws.row_dimensions[r].height = 22

ws.row_dimensions[1].height = 32
ws.row_dimensions[2].height = 24

# Freeze panes
ws.freeze_panes = "A4"

# Hide gridlines
ws.sheet_view.showGridLines = False

# Final save
wb.save(OUTPUT_FILE)

print(f"Report saved: {OUTPUT_FILE}")