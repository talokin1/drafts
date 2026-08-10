import numpy as np
import pandas as pd

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import DataBarRule
from openpyxl.utils import range_boundaries


def create_golden_report(
    model,
    X_train,
    valid_pool,
    inference_result,
    metrics,
    cm,
    BEST_THRESHOLD,
    output_path="Golden_Model_Report.xlsx"
):
    # =========================
    # COLORS / STYLES
    # =========================
    NAVY = "1F4E78"
    BLUE = "5B9BD5"
    LIGHT_BLUE = "DDEBF7"
    LIGHTER_BLUE = "EAF2F8"
    GREEN = "E2F0D9"
    RED = "F4CCCC"
    ORANGE = "FCE4D6"
    WHITE = "FFFFFF"
    GRAY = "D9E1F2"

    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def value_for_excel(x):
        if pd.isna(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x

    def style_range(ws, cell_range, fill=None, font=None, alignment=None, add_border=False):
        min_col, min_row, max_col, max_row = range_boundaries(cell_range)

        for row in ws.iter_rows(
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col
        ):
            for cell in row:
                if fill:
                    cell.fill = PatternFill("solid", fgColor=fill)
                if font:
                    cell.font = font
                if alignment:
                    cell.alignment = alignment
                if add_border:
                    cell.border = border

    def section(ws, cell_range, text, fill=NAVY, font_color=WHITE, size=12):
        style_range(
            ws,
            cell_range,
            fill=fill,
            font=Font(bold=True, color=font_color, size=size),
            alignment=Alignment(horizontal="center", vertical="center", wrap_text=True)
        )

        ws.merge_cells(cell_range)

        min_col, min_row, _, _ = range_boundaries(cell_range)
        ws.cell(min_row, min_col).value = text

    def textbox(ws, cell_range, text, fill=LIGHTER_BLUE, center=False):
        style_range(
            ws,
            cell_range,
            fill=fill,
            alignment=Alignment(
                horizontal="center" if center else "left",
                vertical="center",
                wrap_text=True
            )
        )

        ws.merge_cells(cell_range)

        min_col, min_row, _, _ = range_boundaries(cell_range)
        ws.cell(min_row, min_col).value = text

    def write_table(ws, start_row, start_col, headers, rows):
        for j, header in enumerate(headers, start_col):
            cell = ws.cell(start_row, j, header)
            cell.fill = PatternFill("solid", fgColor=NAVY)
            cell.font = Font(bold=True, color=WHITE)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border

        for i, row in enumerate(rows, start_row + 1):
            for j, val in enumerate(row, start_col):
                cell = ws.cell(i, j, value_for_excel(val))
                cell.border = border
                cell.alignment = Alignment(vertical="center")

                if (i - start_row) % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor=LIGHTER_BLUE)

    # =========================
    # WORKBOOK
    # =========================
    wb = Workbook()
    wb.remove(wb.active)

    # ============================================================
    # 1. MODEL DESCRIPTION
    # ============================================================
    ws = wb.create_sheet("Model description")
    ws.sheet_view.showGridLines = False

    section(ws, "B2:H2", "Task description")

    textbox(
        ws,
        "B3:H6",
        "The goal of the model is to estimate how similar an HNWI client's "
        "current profile is to Golden clients. The model ranks currently "
        "non-Golden HNWI clients by their Golden propensity."
    )

    section(ws, "B8:H8", "Target description")

    textbox(
        ws,
        "B9:H11",
        "TARGET = 1 (Golden): PACKAGE is mandatory and at least two of the "
        "four additional Golden criteria must be satisfied.\n\n"
        "TARGET = 0 (Not Golden): the client does not satisfy the Golden rule."
    )

    write_table(
        ws,
        13,
        2,
        ["Criterion", "Condition"],
        [
            ["PACKAGE", "Mandatory"],
            ["TOTAL_PORTFOLIO", "> 4,000,000"],
            ["LIABILITIES_UAH", "> 1,000,000"],
            ["INCOME(COM+INTEREST)", "> 15,000"],
            ["AMT_DEB_CARD", "> 50,000"]
        ]
    )

    section(ws, "B21:H21", "Model output")

    write_table(
        ws,
        22,
        2,
        ["Output", "Description"],
        [
            ["GOLDEN_SCORE", "Continuous Golden propensity score from 0 to 1"],
            ["MODEL_CLASS", "Golden / Not Golden according to selected threshold"],
            ["GOLDEN_RANK", "Ranking of non-Golden clients by Golden propensity"]
        ]
    )

    ws.column_dimensions["B"].width = 28
    ws.column_dimensions["C"].width = 40

    for col in ["D", "E", "F", "G", "H"]:
        ws.column_dimensions[col].width = 16

    # ============================================================
    # 2. ARCHITECTURE
    # ============================================================
    ws = wb.create_sheet("Architecture")
    ws.sheet_view.showGridLines = False

    section(ws, "B2:G2", "Architecture CatBoost")

    textbox(
        ws,
        "B4:G8",
        "Current client characteristics are fed into the model. "
        "CatBoost estimates a continuous Golden propensity score. "
        "The selected threshold converts this score into the final "
        "Golden / Not Golden classification.",
        center=True
    )

    section(ws, "B10:G11", "INPUT DATA\nHNWI client × features")

    ws.merge_cells("B12:G12")
    ws["B12"] = "↓"
    ws["B12"].font = Font(size=20, bold=True, color=NAVY)
    ws["B12"].alignment = Alignment(horizontal="center")

    section(ws, "B13:G14", "CatBoostClassifier\nBinary classification", BLUE)

    ws.merge_cells("B15:G15")
    ws["B15"] = "↓"
    ws["B15"].font = Font(size=20, bold=True, color=NAVY)
    ws["B15"].alignment = Alignment(horizontal="center")

    section(ws, "B16:G17", "GOLDEN SCORE\nContinuous propensity from 0 to 1")

    ws.merge_cells("B18:G18")
    ws["B18"] = "↓"
    ws["B18"].font = Font(size=20, bold=True, color=NAVY)
    ws["B18"].alignment = Alignment(horizontal="center")

    section(
        ws,
        "B19:G20",
        f"CLASSIFICATION THRESHOLD\n{BEST_THRESHOLD:.2%}",
        ORANGE,
        NAVY
    )

    ws.merge_cells("B21:G21")
    ws["B21"] = "↓"
    ws["B21"].font = Font(size=20, bold=True, color=NAVY)
    ws["B21"].alignment = Alignment(horizontal="center")

    section(ws, "B22:D23", "NOT GOLDEN", RED, NAVY)
    section(ws, "E22:G23", "GOLDEN", GREEN, NAVY)

    for col in ["B", "C", "D", "E", "F", "G"]:
        ws.column_dimensions[col].width = 14

    # ============================================================
    # 3. EXAMPLE
    # ============================================================
    ws = wb.create_sheet("Example")
    ws.sheet_view.showGridLines = False

    example = inference_result.copy()

    if "MODEL_CLASS" not in example.columns:
        example["MODEL_CLASS"] = np.where(
            example["GOLDEN_SCORE"] >= BEST_THRESHOLD,
            "Golden",
            "Not Golden"
        )

    if "GOLDEN_RANK" not in example.columns:
        example = example.sort_values("GOLDEN_SCORE", ascending=False)
        example["GOLDEN_RANK"] = np.arange(1, len(example) + 1)

    example = example[
        ["CONTRAGENTID", "GOLDEN_SCORE", "MODEL_CLASS", "GOLDEN_RANK"]
    ].head(10)

    write_table(
        ws,
        2,
        2,
        ["CLIENT_ID", "Golden propensity", "Model class", "Rank"],
        example.values.tolist()
    )

    for row in range(3, len(example) + 3):
        ws.cell(row, 3).number_format = "0.00%"

    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 18
    ws.column_dimensions["E"].width = 12

    # ============================================================
    # 4. MODEL RESULT
    # ============================================================
    ws = wb.create_sheet("Model result")
    ws.sheet_view.showGridLines = False

    section(ws, "B2:H2", "Model result")

    metric_rows = [
        ["Best threshold", float(BEST_THRESHOLD)],
        ["Precision", float(metrics["Precision"])],
        ["Recall", float(metrics["Recall"])],
        ["F1", float(metrics["F1"])],
        ["ROC-AUC", float(metrics["ROC_AUC"])]
    ]

    write_table(ws, 4, 2, ["Metric", "Value"], metric_rows)

    for row in range(5, 10):
        ws.cell(row, 3).number_format = "0.00%"

    section(ws, "E4:G4", "Confusion matrix")

    cm_array = np.asarray(cm)

    cm_rows = [
        ["", "Pred Not Golden", "Pred Golden"],
        ["Actual Not Golden", int(cm_array[0, 0]), int(cm_array[0, 1])],
        ["Actual Golden", int(cm_array[1, 0]), int(cm_array[1, 1])]
    ]

    for i, row in enumerate(cm_rows, 5):
        for j, value in enumerate(row, 5):
            cell = ws.cell(i, j, value)
            cell.border = border
            cell.alignment = Alignment(horizontal="center", vertical="center")

            if i == 5:
                cell.fill = PatternFill("solid", fgColor=NAVY)
                cell.font = Font(bold=True, color=WHITE)
            elif j == 5:
                cell.fill = PatternFill("solid", fgColor=LIGHT_BLUE)
                cell.font = Font(bold=True)

    textbox(
        ws,
        "B12:H14",
        "Metrics are calculated using out-of-fold predictions. "
        "The classification threshold is selected by the maximum F1 score."
    )

    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 16

    for col in ["E", "F", "G"]:
        ws.column_dimensions[col].width = 20

    # ============================================================
    # 5. FEATURE IMPORTANCE
    # ============================================================
    ws = wb.create_sheet("Feature importance")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "B3"

    fi = model.get_feature_importance(
        valid_pool,
        type="PredictionValuesChange",
        prettified=True
    )

    fi = fi.iloc[:, :2].copy()
    fi.columns = ["Feature", "Importance"]

    total_importance = fi["Importance"].sum()
    fi["Importance_%"] = fi["Importance"] / total_importance

    write_table(
        ws,
        2,
        2,
        ["Feature", "Importance", "Importance_%"],
        fi.values.tolist()
    )

    last_row = len(fi) + 2

    for row in range(3, last_row + 1):
        ws.cell(row, 3).number_format = "0.000"
        ws.cell(row, 4).number_format = "0.000%"

    max_importance = max(float(fi["Importance"].max()), 1)

    ws.conditional_formatting.add(
        f"C3:C{last_row}",
        DataBarRule(
            start_type="num",
            start_value=0,
            end_type="num",
            end_value=max_importance,
            color="FF5B9BD5",
            showValue=True
        )
    )

    ws.conditional_formatting.add(
        f"D3:D{last_row}",
        DataBarRule(
            start_type="num",
            start_value=0,
            end_type="num",
            end_value=1,
            color="FF70AD47",
            showValue=True
        )
    )

    ws.column_dimensions["B"].width = 40
    ws.column_dimensions["C"].width = 18
    ws.column_dimensions["D"].width = 18

    # ============================================================
    # 6. FEATURE DESCRIBE
    # ============================================================
    ws = wb.create_sheet("Feature describe")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "B3"

    feature_rows = []

    for col in X_train.columns:
        feature_rows.append([
            col,
            str(X_train[col].dtype),
            float(X_train[col].isna().mean()),
            int(X_train[col].nunique(dropna=False)),
            ""
        ])

    write_table(
        ws,
        2,
        2,
        ["Feature", "Dtype", "Missing_%", "N_unique", "Description"],
        feature_rows
    )

    last_row = len(feature_rows) + 2

    for row in range(3, last_row + 1):
        ws.cell(row, 4).number_format = "0.00%"

    ws.column_dimensions["B"].width = 40
    ws.column_dimensions["C"].width = 18
    ws.column_dimensions["D"].width = 15
    ws.column_dimensions["E"].width = 15
    ws.column_dimensions["F"].width = 45


    wb.save(output_path)
    print(f"Report saved: {output_path}")


create_golden_report(
    model=model,
    X_train=X_train,
    valid_pool=valid_pool,
    inference_result=inference_result,
    metrics=metrics,
    cm=cm,
    BEST_THRESHOLD=BEST_THRESHOLD,
    output_path="Golden_Model_Report.xlsx"
)