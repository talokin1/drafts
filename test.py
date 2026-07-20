from sklearn.model_selection import train_test_split

df = model_dataset.copy()

df = df[~df["RECORD_TYPE"].isin(["BAD_RECORD", "BAD"])].copy()


def parse_products(value):
    if pd.isna(value):
        return {"NOTHING_TO_DO"}

    products = {
        x.strip().upper()
        for x in str(value).split(",")
        if x.strip()
    }

    return products or {"NOTHING_TO_DO"}


df["TARGET_SET"] = df["ACTUAL_PRODUCT"].apply(parse_products)

df["STRATIFY_TARGET"] = df["TARGET_SET"].apply(
    lambda x: ",".join(sorted(x))
)

counts = df["STRATIFY_TARGET"].value_counts()

df["STRATIFY_GROUP"] = df["STRATIFY_TARGET"].where(
    df["STRATIFY_TARGET"].map(counts) >= 5,
    "OTHER"
)

group_counts = df["STRATIFY_GROUP"].value_counts()
single_mask = df["STRATIFY_GROUP"].map(group_counts) < 2

df_main = df[~single_mask].copy()
df_single = df[single_mask].copy()

train, test = train_test_split(
    df_main,
    test_size=0.20,
    random_state=42,
    stratify=df_main["STRATIFY_GROUP"]
)

# Одиничні випадки додаємо тільки в train
train = pd.concat(
    [train, df_single],
    ignore_index=True
)












from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRODUCTS = ["LIABILITIES", "ASSETS", "FX"]
SCORE_COLS = {
    "LIABILITIES": "LIAB_PRIMARY",
    "ASSETS": "ASSETS_PRIMARY",
    "FX": "FX_PRIMARY",
}


def _as_set(value) -> set[str]:
    if isinstance(value, set):
        result = {str(x).strip().upper() for x in value}
    elif isinstance(value, (list, tuple, np.ndarray)):
        result = {str(x).strip().upper() for x in value}
    elif pd.isna(value):
        result = set()
    else:
        result = {x.strip().upper() for x in str(value).split(",") if x.strip()}
    return result or {"NOTHING_TO_DO"}


def _join_set(value) -> str:
    return ", ".join(sorted(_as_set(value)))


def generate_model_report(
    test: pd.DataFrame,
    product_metrics: pd.DataFrame,
    best: dict,
    output_path: str | Path = "corp_recommendation_model_report.xlsx",
    example_rows: int = 15,
) -> str:
    """Generate an Excel report for the two-stage corporate recommendation model.

    Required columns in test:
      TARGET_SET or ACTUAL_PRODUCT,
      RECOMMENDED_PRODUCT,
      CONTRAGENTID / IDENTIFYCODE (optional),
      product scores and/or *_PCT columns.
    """
    df = test.copy()

    if "TARGET_SET" in df:
        df["ACTUAL_SET"] = df["TARGET_SET"].apply(_as_set)
    else:
        df["ACTUAL_SET"] = df["ACTUAL_PRODUCT"].apply(_as_set)

    df["PRED_SET"] = df["RECOMMENDED_PRODUCT"].apply(_as_set)
    df["ACTUAL_PRODUCT_REPORT"] = df["ACTUAL_SET"].apply(_join_set)
    df["RECOMMENDED_PRODUCT_REPORT"] = df["PRED_SET"].apply(_join_set)

    pct_cols = [f"{p}_PCT" for p in PRODUCTS]
    available_pct = [c for c in pct_cols if c in df.columns]
    if "ACTION_SCORE" not in df.columns:
        if available_pct:
            df["ACTION_SCORE"] = df[available_pct].max(axis=1)
        else:
            raw = [SCORE_COLS[p] for p in PRODUCTS if SCORE_COLS[p] in df.columns]
            df["ACTION_SCORE"] = df[raw].max(axis=1)

    actual_action = df["ACTUAL_SET"].apply(lambda x: x != {"NOTHING_TO_DO"})
    pred_action = df["PRED_SET"].apply(lambda x: x != {"NOTHING_TO_DO"})
    product_hit = [bool(a & p - {"NOTHING_TO_DO"}) for a, p in zip(df["ACTUAL_SET"], df["PRED_SET"])]

    tp = int((actual_action & pred_action).sum())
    fp = int((~actual_action & pred_action).sum())
    fn = int((actual_action & ~pred_action).sum())
    tn = int((~actual_action & ~pred_action).sum())

    action_precision = tp / max(tp + fp, 1)
    action_recall = tp / max(tp + fn, 1)
    action_f1 = 2 * action_precision * action_recall / max(action_precision + action_recall, 1e-12)
    nothing_recall = tn / max(tn + fp, 1)
    exact_match = np.mean([a == p for a, p in zip(df["ACTUAL_SET"], df["PRED_SET"])])
    hit_rate = np.mean([bool(a & p) for a, p in zip(df["ACTUAL_SET"], df["PRED_SET"])])
    positive_hit = np.mean(np.array(product_hit)[actual_action.to_numpy()]) if actual_action.any() else 0

    thresholds_raw = best.get("thresholds", {})
    if isinstance(thresholds_raw, dict):
        thresholds = {p: float(thresholds_raw.get(p, np.nan)) for p in PRODUCTS}
    else:
        thresholds = dict(zip(PRODUCTS, np.asarray(thresholds_raw, dtype=float)))

    gate = float(best.get("gate", best.get("gate_threshold", np.nan)))
    tie_delta = float(best.get("tie_delta", np.nan))

    overall_metrics = pd.DataFrame(
        {
            "Metric": [
                "Exact match accuracy", "Hit rate", "ACTION precision",
                "ACTION recall", "ACTION F1", "Positive hit rate",
                "NOTHING recall", "Product Macro F1",
            ],
            "Value": [
                exact_match, hit_rate, action_precision, action_recall,
                action_f1, positive_hit, nothing_recall,
                float(product_metrics["F1"].mean()),
            ],
        }
    )

    confusion = pd.DataFrame(
        {
            "Indicator": [
                "All customers", "Actual ACTION customers", "Predicted ACTION customers",
                "Correctly found", "False positives", "Missed positives", "True negatives",
            ],
            "Value": [len(df), int(actual_action.sum()), int(pred_action.sum()), tp, fp, fn, tn],
        }
    )

    product_table = product_metrics.copy()
    if "Product" not in product_table.columns:
        product_table = product_table.reset_index().rename(columns={product_table.index.name or "index": "Product"})
    rec_counts = df.loc[pred_action, "PRED_SET"].explode().value_counts()
    product_table["Recommendations"] = product_table["Product"].map(rec_counts).fillna(0).astype(int)
    product_table["Threshold"] = product_table["Product"].map(thresholds)

    if "ACTION_SCORE" in df.columns:
        bins = np.linspace(0, 1, 11)
        df["SCORE_BUCKET"] = pd.cut(df["ACTION_SCORE"].clip(0, 1), bins=bins, include_lowest=True)
        bucket = (
            df.assign(ACTUAL_ACTION=actual_action.astype(int), PRED_ACTION=pred_action.astype(int), HIT=product_hit)
            .groupby("SCORE_BUCKET", observed=False)
            .agg(
                Customers=("ACTUAL_ACTION", "size"),
                Actual_ACTION=("ACTUAL_ACTION", "sum"),
                Predicted_ACTION=("PRED_ACTION", "sum"),
                Correct_product=("HIT", "sum"),
                Avg_ACTION_score=("ACTION_SCORE", "mean"),
            )
            .reset_index()
        )
        bucket["Actual_ACTION_rate"] = bucket["Actual_ACTION"] / bucket["Customers"].replace(0, np.nan)
        bucket = bucket.iloc[::-1].reset_index(drop=True)
    else:
        bucket = pd.DataFrame()

    id_cols = [c for c in ["CONTRAGENTID", "IDENTIFYCODE"] if c in df.columns]
    score_cols = [c for c in [
        "LIAB_PRIMARY", "ASSETS_PRIMARY", "FX_PRIMARY",
        "LIABILITIES_PCT", "ASSETS_PCT", "FX_PCT", "ACTION_SCORE",
    ] if c in df.columns]
    example = df.sort_values("ACTION_SCORE", ascending=False).head(example_rows)
    example = example[id_cols + score_cols + ["ACTUAL_PRODUCT_REPORT", "RECOMMENDED_PRODUCT_REPORT"]]
    example = example.rename(columns={
        "ACTUAL_PRODUCT_REPORT": "ACTUAL_PRODUCT",
        "RECOMMENDED_PRODUCT_REPORT": "RECOMMENDED_PRODUCT",
    })

    feature_desc = pd.DataFrame(
        [
            ["CONTRAGENTID", "Internal customer identifier", "Identifier"],
            ["IDENTIFYCODE", "Eight-digit customer identification code", "Identifier"],
            ["LIAB_PRIMARY", "Propensity score for liabilities", "Input score"],
            ["ASSETS_PRIMARY", "Propensity score for assets", "Input score"],
            ["FX_PRIMARY", "Propensity score for FX", "Input score"],
            ["*_PCT", "Score percentile calculated from the train distribution", "Transformation"],
            ["ACTION_SCORE", "Maximum normalized product score", "Stage 1"],
            ["ACTUAL_PRODUCT", "Product actually activated; NOTHING_TO_DO means no product", "Target"],
            ["RECORD_TYPE", "Quality/type of the observation", "Control field"],
            ["RECOMMENDED_PRODUCT", "Final one- or two-product recommendation", "Output"],
        ],
        columns=["Field", "Description", "Role"],
    )

    path = str(Path(output_path))
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        wb = writer.book
        navy = "#1F4E78"
        blue = "#4472C4"
        light_blue = "#D9EAF7"
        pale_blue = "#EAF2F8"
        green = "#70AD47"
        pale_green = "#E2F0D9"
        red = "#C00000"
        orange = "#C65911"
        pale_orange = "#FCE4D6"
        gray = "#F2F2F2"
        white = "#FFFFFF"

        fmt_title = wb.add_format({"bold": True, "font_color": white, "bg_color": navy, "align": "center", "valign": "vcenter", "font_size": 12})
        fmt_sub = wb.add_format({"bold": True, "font_color": navy, "bg_color": light_blue, "valign": "vcenter", "font_size": 11})
        fmt_text = wb.add_format({"text_wrap": True, "valign": "vcenter", "font_size": 11})
        fmt_note = wb.add_format({"italic": True, "bold": True, "font_color": navy, "bg_color": pale_blue, "text_wrap": True})
        fmt_header = wb.add_format({"bold": True, "font_color": white, "bg_color": navy, "align": "center", "border": 1})
        fmt_pct = wb.add_format({"num_format": "0.00%", "border": 1})
        fmt_num = wb.add_format({"num_format": "0", "border": 1})
        fmt_dec = wb.add_format({"num_format": "0.000", "border": 1})
        fmt_cell = wb.add_format({"border": 1})

        def section(ws, row: int, title: str, end_col: int = 9, color: str = navy) -> int:
            f = wb.add_format({"bold": True, "font_color": white, "bg_color": color, "align": "center", "valign": "vcenter", "font_size": 12})
            ws.merge_range(row, 0, row, end_col, title, f)
            ws.set_row(row, 22)
            return row + 1

        # Model description
        ws = wb.add_worksheet("Model description")
        writer.sheets["Model description"] = ws
        ws.hide_gridlines(2)
        ws.set_column("A:J", 18)
        row = section(ws, 1, "Task description")
        ws.merge_range(row, 0, row + 2, 9,
            "The model recommends the next best corporate banking product: LIABILITIES, ASSETS, FX, or NOTHING_TO_DO. "
            "The decision is based on product propensity scores and is optimized on a representative train/test split.", fmt_text)
        row += 4
        row = section(ws, row, "Target description")
        ws.merge_range(row, 0, row + 2, 9,
            "The target is the product actually activated by the customer. Multiple products may be present. "
            "NOTHING_TO_DO means that no product was activated and the customer should not receive a recommendation.", fmt_text)
        row += 4
        row = section(ws, row, "The model works in two stages")
        ws.merge_range(row, 0, row, 9, "1. ACTION / NOTHING_TO_DO gate", fmt_sub)
        row += 1
        ws.merge_range(row, 0, row + 1, 9,
            "Product scores are transformed into train-based percentiles. ACTION_SCORE is the maximum percentile. "
            f"If the normalized margin is below the gate ({gate:.3f}), the result is NOTHING_TO_DO.", fmt_text)
        row += 3
        ws.write(row, 0, "TARGET = 1", wb.add_format({"bold": True, "font_color": white, "bg_color": green, "align": "center"}))
        ws.merge_range(row, 1, row, 9, "At least one product was actually activated.", wb.add_format({"bg_color": pale_green}))
        row += 1
        ws.write(row, 0, "TARGET = 0", wb.add_format({"bold": True, "font_color": white, "bg_color": red, "align": "center"}))
        ws.merge_range(row, 1, row, 9, "No product was activated: NOTHING_TO_DO.", wb.add_format({"bg_color": gray}))
        row += 2
        ws.merge_range(row, 0, row, 9, "2. Product selection", wb.add_format({"bold": True, "font_color": orange, "bg_color": pale_orange}))
        row += 1
        ws.merge_range(row, 0, row + 2, 9,
            "For ACTION customers, each product must pass its own threshold. Products are ranked by normalized margin above threshold. "
            f"A second product is added only when the difference between the two strongest margins is no more than {tie_delta:.3f}.", fmt_text)
        row += 4
        ws.merge_range(row, 0, row, 9, "The final report contains the recommendation, model quality, thresholds and examples.", fmt_title)
        ws.set_row(row, 24)

        # Architecture
        ws = wb.add_worksheet("Architecture")
        writer.sheets["Architecture"] = ws
        ws.hide_gridlines(2)
        ws.set_column("A:A", 4)
        ws.set_column("B:F", 18)
        ws.set_column("G:J", 14)
        section(ws, 1, "Two-stage rule-based architecture", 5)
        ws.merge_range("B3:F6", "One customer is represented by three product propensity scores. "
                       "All transformations and thresholds are estimated only on train data.", fmt_text)
        blocks = [
            (8, "INPUT DATA\nLIAB_PRIMARY | ASSETS_PRIMARY | FX_PRIMARY", navy),
            (11, "PERCENTILE TRANSFORMATION\nTrain distribution for each product", "#5B9BD5"),
            (14, f"STAGE 1: ACTION GATE\nACTION_SCORE = max(product percentiles)\nGate = {gate:.3f}", blue),
            (18, "STAGE 2: PRODUCT THRESHOLDS\n" + " | ".join(f"{p}: {thresholds[p]:.3f}" for p in PRODUCTS), blue),
            (22, f"TIE RULE\nAdd top-2 when margin difference ≤ {tie_delta:.3f}", blue),
            (26, "OUTPUT\nLIABILITIES / ASSETS / FX / NOTHING_TO_DO", navy),
        ]
        for i, (r, text, color) in enumerate(blocks):
            ws.merge_range(r, 1, r + 1, 5, text, wb.add_format({"bold": True, "font_color": white, "bg_color": color, "align": "center", "valign": "vcenter", "text_wrap": True, "border": 1}))
            if i < len(blocks) - 1:
                ws.merge_range(r + 2, 2, r + 2, 4, "↓", wb.add_format({"bold": True, "font_size": 20, "font_color": navy, "align": "center"}))

        # Example
        example.to_excel(writer, sheet_name="Example", startrow=1, startcol=0, index=False)
        ws = writer.sheets["Example"]
        ws.hide_gridlines(2)
        ws.set_row(1, 22)
        ws.set_column(0, len(example.columns) - 1, 18)
        for c, col in enumerate(example.columns):
            ws.write(1, c, col, fmt_header)
        for c, col in enumerate(example.columns):
            if col.endswith("_PCT") or col == "ACTION_SCORE":
                ws.set_column(c, c, 16, wb.add_format({"num_format": "0.00%"}))
            elif col.endswith("PRIMARY"):
                ws.set_column(c, c, 16, wb.add_format({"num_format": "0.000"}))
        ws.autofilter(1, 0, len(example) + 1, len(example.columns) - 1)
        ws.freeze_panes(2, 0)

        # Model result
        ws = wb.add_worksheet("Model result")
        writer.sheets["Model result"] = ws
        ws.hide_gridlines(2)
        ws.set_column("A:A", 22)
        ws.set_column("B:H", 16)
        ws.set_column("J:J", 28)
        ws.set_column("K:K", 16)
        section(ws, 0, "Model results", 7)
        if not bucket.empty:
            bucket.to_excel(writer, sheet_name="Model result", startrow=2, startcol=0, index=False)
            for c, col in enumerate(bucket.columns):
                ws.write(2, c, col, fmt_header)
            start = 3
            end = start + len(bucket) - 1
            for c, col in enumerate(bucket.columns):
                if col in ["Avg_ACTION_score", "Actual_ACTION_rate"]:
                    ws.set_column(c, c, 18, fmt_pct)
                else:
                    ws.set_column(c, c, 18, fmt_num if col != "SCORE_BUCKET" else fmt_cell)
            ws.conditional_format(start, 4, end, 4, {"type": "data_bar", "bar_color": "#5B9BD5"})
            ws.conditional_format(start, 5, end, 5, {"type": "data_bar", "bar_color": "#70AD47"})

        ws.merge_range("J1:K1", "Dashboard", fmt_title)
        confusion.to_excel(writer, sheet_name="Model result", startrow=1, startcol=9, index=False, header=False)
        for r in range(1, 1 + len(confusion)):
            ws.write(r, 9, confusion.iloc[r - 1, 0], wb.add_format({"bold": True, "font_color": navy, "bg_color": light_blue}))
            ws.write_number(r, 10, int(confusion.iloc[r - 1, 1]))

        metric_row = max(12, len(confusion) + 3)
        ws.merge_range(metric_row, 9, metric_row, 10, "Quality metrics", fmt_title)
        overall_metrics.to_excel(writer, sheet_name="Model result", startrow=metric_row + 1, startcol=9, index=False, header=False)
        for r in range(metric_row + 1, metric_row + 1 + len(overall_metrics)):
            ws.write(r, 9, overall_metrics.iloc[r - metric_row - 1, 0], wb.add_format({"bold": True, "font_color": navy, "bg_color": light_blue}))
            ws.write_number(r, 10, float(overall_metrics.iloc[r - metric_row - 1, 1]), wb.add_format({"num_format": "0.00%"}))

        product_start = max(18, len(bucket) + 5)
        ws.merge_range(product_start, 0, product_start, 7, "Product-level metrics", fmt_title)
        product_table.to_excel(writer, sheet_name="Model result", startrow=product_start + 1, startcol=0, index=False)
        for c, col in enumerate(product_table.columns):
            ws.write(product_start + 1, c, col, fmt_header)
            if col in ["Precision", "Recall", "F1"]:
                ws.set_column(c, c, 15, fmt_pct)
            elif col == "Threshold":
                ws.set_column(c, c, 15, fmt_dec)
            else:
                ws.set_column(c, c, 18)
        data_first = product_start + 2
        data_last = data_first + len(product_table) - 1
        for c in [product_table.columns.get_loc(x) for x in ["Precision", "Recall", "F1"] if x in product_table.columns]:
            ws.conditional_format(data_first, c, data_last, c, {"type": "data_bar", "bar_color": "#5B9BD5"})

        # Feature importance / rule importance
        rule_table = product_table[[c for c in ["Product", "Threshold", "Precision", "Recall", "F1", "Support", "Recommendations"] if c in product_table.columns]].copy()
        rule_table.to_excel(writer, sheet_name="Feature importance", startrow=3, startcol=0, index=False)
        ws = writer.sheets["Feature importance"]
        ws.hide_gridlines(2)
        ws.set_column("A:G", 19)
        section(ws, 0, "Rule importance and decision quality", 6)
        ws.merge_range("A2:G2", "This is a rule-based model; therefore classical feature importance is not applicable. "
                       "The table shows the score components, thresholds and observed quality.", fmt_note)
        for c, col in enumerate(rule_table.columns):
            ws.write(3, c, col, fmt_header)
        for c, col in enumerate(rule_table.columns):
            if col in ["Precision", "Recall", "F1"]:
                ws.set_column(c, c, 18, fmt_pct)
            elif col == "Threshold":
                ws.set_column(c, c, 18, fmt_dec)
        ws.conditional_format(4, 2, 3 + len(rule_table), 4, {"type": "data_bar", "bar_color": "#70AD47"})

        # Feature description
        feature_desc.to_excel(writer, sheet_name="Feature describe", startrow=2, startcol=0, index=False)
        ws = writer.sheets["Feature describe"]
        ws.hide_gridlines(2)
        ws.set_column("A:A", 24)
        ws.set_column("B:B", 65)
        ws.set_column("C:C", 20)
        section(ws, 0, "Feature description", 2)
        for c, col in enumerate(feature_desc.columns):
            ws.write(2, c, col, fmt_header)
        ws.freeze_panes(3, 0)

    return path


# Example call after model evaluation:
# report_path = generate_model_report(test, product_metrics, best)
# print(report_path)



from generate_corp_rec_report import generate_model_report

report_path = generate_model_report(
    test=test,
    product_metrics=product_metrics,
    best=best,
    output_path="corp_recommendation_model_report.xlsx"
)

print(report_path)