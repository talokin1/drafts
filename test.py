import numpy as np
import pandas as pd
import xlsxwriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def _excel_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def _model_name(model):
    if model is None:
        return "Binary classifier"
    m = model[0] if isinstance(model, (list, tuple)) and model else model
    name = type(m).__name__
    aliases = {"LGBMClassifier": "LightGBM", "CatBoostClassifier": "CatBoost", "XGBClassifier": "XGBoost"}
    return aliases.get(name, name)


def _feature_importance(model, feature_names):
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    values = []
    for m in models:
        if m is None:
            continue
        imp = None
        if hasattr(m, "booster_") and hasattr(m.booster_, "feature_importance"):
            imp = np.asarray(m.booster_.feature_importance(importance_type="gain"), dtype=float)
        elif hasattr(m, "get_feature_importance"):
            imp = np.asarray(m.get_feature_importance(), dtype=float)
        elif hasattr(m, "feature_importances_"):
            imp = np.asarray(m.feature_importances_, dtype=float)
        if imp is not None and len(imp) == len(feature_names):
            values.append(imp)

    if not values:
        return pd.DataFrame({"Feature": feature_names, "Importance": np.nan, "Importance_%": np.nan})

    imp = np.mean(values, axis=0)
    total = imp.sum()
    out = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    out["Importance_%"] = out["Importance"] / total if total > 0 else np.nan
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def _prepare_feature_description(feature_names, X_reference=None, feature_descriptions=None):
    dtypes = {c: str(X_reference[c].dtype) for c in feature_names if X_reference is not None and c in X_reference.columns}

    if feature_descriptions is None:
        descriptions = {}
    elif isinstance(feature_descriptions, dict):
        descriptions = feature_descriptions
    elif isinstance(feature_descriptions, pd.DataFrame):
        tmp = feature_descriptions.copy()
        feature_col = next((c for c in tmp.columns if c.lower() in {"feature", "feature_name", "column", "variable"}), tmp.columns[0])
        desc_col = next((c for c in tmp.columns if c.lower() in {"description", "feature_description", "describe"}), None)
        descriptions = dict(zip(tmp[feature_col], tmp[desc_col])) if desc_col else {}
    else:
        raise TypeError("feature_descriptions must be dict, DataFrame or None")

    return pd.DataFrame({
        "Feature": feature_names,
        "Data type": [dtypes.get(c, "") for c in feature_names],
        "Description": [descriptions.get(c, "") for c in feature_names],
    })


def _make_formats(workbook):
    navy = "#1F4E78"
    blue = "#5B9BD5"
    light_blue = "#D9EAF7"
    very_light_blue = "#EAF2F8"
    green = "#548235"
    light_green = "#E2F0D9"
    red = "#C00000"
    light_red = "#FCE4D6"
    orange = "#ED7D31"
    light_orange = "#FCE4D6"
    border = "#B7C9D6"

    return {
        "title": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "align": "center", "valign": "vcenter", "font_size": 12}),
        "section": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "align": "center", "valign": "vcenter", "font_size": 11}),
        "body": workbook.add_format({"bg_color": very_light_blue, "font_color": "#000000", "align": "left", "valign": "vcenter", "text_wrap": True, "border": 0}),
        "sub_blue": workbook.add_format({"bold": True, "font_color": "#17365D", "bg_color": light_blue, "align": "left", "valign": "vcenter"}),
        "sub_orange": workbook.add_format({"bold": True, "font_color": orange, "bg_color": light_orange, "align": "left", "valign": "vcenter"}),
        "target1": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": green, "align": "center", "valign": "vcenter"}),
        "target0": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": red, "align": "center", "valign": "vcenter"}),
        "target1_desc": workbook.add_format({"bg_color": light_green, "align": "left", "valign": "vcenter"}),
        "target0_desc": workbook.add_format({"bg_color": "#F2F2F2", "align": "left", "valign": "vcenter"}),
        "header": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "align": "center", "valign": "vcenter", "border": 0}),
        "cell": workbook.add_format({"align": "left", "valign": "vcenter", "border": 1, "border_color": border}),
        "cell_center": workbook.add_format({"align": "center", "valign": "vcenter", "border": 1, "border_color": border}),
        "integer": workbook.add_format({"num_format": "0", "align": "center", "valign": "vcenter", "border": 1, "border_color": border}),
        "float": workbook.add_format({"num_format": "0.000000", "align": "right", "valign": "vcenter", "border": 1, "border_color": border}),
        "pct": workbook.add_format({"num_format": "0.00%", "align": "right", "valign": "vcenter", "border": 1, "border_color": border}),
        "metric_name": workbook.add_format({"bold": True, "bg_color": light_blue, "align": "left", "valign": "vcenter", "border": 1, "border_color": border}),
        "metric_val": workbook.add_format({"num_format": "0.0000", "align": "center", "valign": "vcenter", "border": 1, "border_color": border}),
        "label": workbook.add_format({"bold": True, "bg_color": light_blue, "align": "left", "valign": "vcenter", "border": 1, "border_color": border}),
        "value": workbook.add_format({"align": "center", "valign": "vcenter", "border": 1, "border_color": border}),
        "cm_header": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "align": "center", "valign": "vcenter", "border": 1, "border_color": "#FFFFFF"}),
        "cm_diag": workbook.add_format({"bold": True, "bg_color": light_green, "align": "center", "valign": "vcenter", "border": 1, "border_color": border}),
        "cm_error": workbook.add_format({"bold": True, "bg_color": light_red, "align": "center", "valign": "vcenter", "border": 1, "border_color": border}),
        "arch_dark": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "align": "center", "valign": "vcenter", "text_wrap": True, "font_size": 11}),
        "arch_mid": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": blue, "align": "center", "valign": "vcenter", "text_wrap": True, "font_size": 11}),
        "arch_light": workbook.add_format({"bg_color": very_light_blue, "align": "center", "valign": "vcenter", "text_wrap": True, "font_size": 11}),
        "arrow": workbook.add_format({"bold": True, "font_color": navy, "align": "center", "valign": "vcenter", "font_size": 22}),
    }


def _write_df(ws, df, start_row, start_col, formats, percent_cols=None, integer_cols=None):
    percent_cols = set(percent_cols or [])
    integer_cols = set(integer_cols or [])
    for j, col in enumerate(df.columns):
        ws.write(start_row, start_col + j, col, formats["header"])
    for i, row in enumerate(df.itertuples(index=False, name=None), start=start_row + 1):
        for j, value in enumerate(row):
            col = df.columns[j]
            fmt = formats["pct"] if col in percent_cols else formats["integer"] if col in integer_cols else formats["cell"]
            ws.write(i, start_col + j, _excel_value(value), fmt)
    return start_row + len(df)


def create_golden_model_report(
    y_true,
    oof_proba,
    inference_df,
    model,
    X_reference=None,
    feature_names=None,
    feature_descriptions=None,
    threshold=0.5,
    output_path="Golden_Model_Report.xlsx",
    id_col="CONTRAGENTID",
    score_col="GOLDEN_SCORE",
    rank_col="GOLDEN_RANK",
    flag_col="HIGH_GOLDEN_PROPENSITY",
    example_rows=10,
):
    y_true = np.asarray(y_true).astype(int)
    oof_proba = np.asarray(oof_proba).astype(float)
    if len(y_true) != len(oof_proba):
        raise ValueError("y_true and oof_proba must have the same length")

    if feature_names is None:
        if X_reference is None:
            raise ValueError("Pass feature_names or X_reference")
        feature_names = list(X_reference.columns)
    else:
        feature_names = list(feature_names)

    y_pred = (oof_proba >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, oof_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = pd.DataFrame({"Metric": ["Precision", "Recall", "F1", "ROC_AUC"], "Value": [precision, recall, f1, roc_auc]})
    importance = _feature_importance(model, feature_names)
    feature_desc = _prepare_feature_description(feature_names, X_reference, feature_descriptions)
    model_name = _model_name(model)

    example_cols = [c for c in [id_col, score_col, rank_col, flag_col] if c in inference_df.columns]
    if not example_cols:
        example_cols = list(inference_df.columns[: min(6, len(inference_df.columns))])
    example = inference_df[example_cols].copy()
    if score_col in example.columns:
        example = example.sort_values(score_col, ascending=False)
    example = example.head(example_rows).reset_index(drop=True)

    wb = xlsxwriter.Workbook(output_path)
    wb.set_properties({"title": "Golden model report", "subject": "Model documentation and OOF results", "author": "Data Science"})
    fmt = _make_formats(wb)

    # 1. Model description
    ws = wb.add_worksheet("Model description")
    ws.hide_gridlines(2)
    ws.set_zoom(95)
    ws.set_column("A:A", 3)
    ws.set_column("B:L", 14)
    ws.set_column("B:B", 24)
    ws.merge_range("B2:L2", "Task description", fmt["section"])
    ws.set_row(1, 22)
    ws.merge_range("B3:L6", "The goal of the model is to estimate the probability that an HNWI client belongs to the Golden segment. The model uses only features available at the prediction moment. The output is used to rank clients by GOLDEN_SCORE and select customers with high Golden propensity.", fmt["body"])
    ws.set_row(2, 26); ws.set_row(3, 26); ws.set_row(4, 26); ws.set_row(5, 26)

    ws.merge_range("B8:L8", "Target description", fmt["section"])
    ws.merge_range("B9:L11", "The target is binary and is built from the current Golden status of the client in the training sample. The model predicts the probability of TARGET = 1.", fmt["body"])

    ws.merge_range("B13:L13", "The model works in three stages", fmt["section"])
    ws.merge_range("B15:L15", "1. Feature preparation", fmt["sub_blue"])
    ws.merge_range("B16:L18", "Client-level features are prepared for the model. Identifiers, the target itself and technical/service columns are not used as predictors. The same feature set and preprocessing must be applied during inference.", fmt["body"])

    ws.merge_range("B20:L20", "2. Forecast of Golden probability", fmt["sub_blue"])
    ws.merge_range("B21:L22", f"The {model_name} classifier estimates P(Golden) for every client. The model is evaluated on out-of-fold predictions, which are not produced by a model trained on the same observation.", fmt["body"])
    ws.merge_range("B24:D24", "TARGET = 1", fmt["target1"])
    ws.merge_range("E24:L24", "Client belongs to the Golden segment.", fmt["target1_desc"])
    ws.merge_range("B25:D25", "TARGET = 0", fmt["target0"])
    ws.merge_range("E25:L25", "HNWI client does not belong to the Golden segment.", fmt["target0_desc"])

    ws.merge_range("B27:L29", f"The output of the classification stage is GOLDEN_SCORE from 0 to 1. The decision threshold used in this report is {threshold:.4f}.", fmt["body"])

    ws.merge_range("B31:L31", "3. Ranking and final selection", fmt["sub_orange"])
    ws.merge_range("B32:L34", "Clients are sorted by GOLDEN_SCORE in descending order. GOLDEN_RANK shows the position in the ranking; HIGH_GOLDEN_PROPENSITY marks clients above the selected threshold for business use.", fmt["body"])

    # 2. Architecture
    ws = wb.add_worksheet("Architecture")
    ws.hide_gridlines(2)
    ws.set_zoom(95)
    ws.set_column("A:A", 3)
    ws.set_column("B:G", 13)
    ws.merge_range("B2:G3", f"Architecture — {model_name}", fmt["title"])
    ws.merge_range("B4:G10", f"One client is represented by a vector of selected behavioral, financial and profile features. The classifier estimates the probability of Golden status. OOF predictions are used for unbiased validation; the trained model is then applied to the full HNWI inference population.", fmt["arch_light"])

    blocks = [
        ("B13:G15", "INPUT DATA\nClient × feature number", "arch_dark"),
        ("B17:G19", "FEATURE PREPARATION\nSelected inference-time features", "arch_mid"),
        ("B21:G23", f"{model_name.upper()} CLASSIFIER\nBinary classification: Golden / Not Golden", "arch_dark"),
        ("B25:G27", "GOLDEN SCORE\nP(Golden) from 0 to 1", "arch_mid"),
        ("B29:G31", "THRESHOLD + RANKING\nHIGH_GOLDEN_PROPENSITY + GOLDEN_RANK", "arch_dark"),
    ]
    for rng, text, style in blocks:
        ws.merge_range(rng, text, fmt[style])
    for r in [15, 19, 23, 27]:
        ws.merge_range(r, 1, r, 6, "↓", fmt["arrow"])
    for r in range(12, 31):
        ws.set_row(r, 22)

    # 3. Example
    ws = wb.add_worksheet("Exaple")
    ws.hide_gridlines(2)
    ws.set_zoom(100)
    ws.set_column("A:A", 3)
    widths = {id_col: 18, score_col: 16, rank_col: 14, flag_col: 24}
    for j, col in enumerate(example.columns, start=1):
        ws.set_column(j, j, widths.get(col, 18))
    percent_cols = [score_col] if score_col in example.columns else []
    integer_cols = [rank_col, flag_col] if rank_col in example.columns and flag_col in example.columns else [c for c in [rank_col, flag_col] if c in example.columns]
    end_row = _write_df(ws, example, 1, 1, fmt, percent_cols=percent_cols, integer_cols=integer_cols)
    ws.freeze_panes(2, 1)
    if flag_col in example.columns and len(example):
        col_idx = example.columns.get_loc(flag_col) + 1
        green_fmt = wb.add_format({"bg_color": "#E2F0D9", "font_color": "#375623", "bold": True, "align": "center"})
        ws.conditional_format(2, col_idx, end_row, col_idx, {"type": "cell", "criteria": "==", "value": 1, "format": green_fmt})

    # 4. Model result
    ws = wb.add_worksheet("Model result")
    ws.hide_gridlines(2)
    ws.set_zoom(100)
    ws.set_column("A:A", 3)
    ws.set_column("B:B", 22)
    ws.set_column("C:C", 16)
    ws.set_column("D:D", 4)
    ws.set_column("E:G", 19)
    ws.merge_range("B2:G2", "OOF MODEL RESULT", fmt["title"])

    for i, (name, value) in enumerate(metrics.itertuples(index=False, name=None), start=4):
        ws.write(i - 1, 1, name, fmt["metric_name"])
        ws.write_number(i - 1, 2, float(value), fmt["metric_val"])
    ws.write("B9", "Decision threshold", fmt["label"]); ws.write_number("C9", float(threshold), fmt["metric_val"])
    ws.write("B10", "OOF observations", fmt["label"]); ws.write_number("C10", int(len(y_true)), fmt["value"])

    ws.merge_range("B12:D12", "Confusion matrix", fmt["section"])
    ws.write("B13", "", fmt["cm_header"]); ws.write("C13", "Pred NOT Golden", fmt["cm_header"]); ws.write("D13", "Pred Golden", fmt["cm_header"])
    ws.write("B14", "Actual NOT Golden", fmt["cm_header"]); ws.write_number("C14", int(tn), fmt["cm_diag"]); ws.write_number("D14", int(fp), fmt["cm_error"])
    ws.write("B15", "Actual Golden", fmt["cm_header"]); ws.write_number("C15", int(fn), fmt["cm_error"]); ws.write_number("D15", int(tp), fmt["cm_diag"])

    chart = wb.add_chart({"type": "column"})
    chart.add_series({"name": "OOF metric", "categories": "='Model result'!$B$4:$B$7", "values": "='Model result'!$C$4:$C$7", "fill": {"color": "#5B9BD5"}, "border": {"none": True}})
    chart.set_title({"name": "OOF metrics"})
    chart.set_y_axis({"min": 0, "max": 1, "major_unit": 0.2, "num_format": "0%"})
    chart.set_legend({"none": True})
    chart.set_style(10)
    chart.set_size({"width": 520, "height": 280})
    ws.insert_chart("E4", chart)

    ws.write("F13", "Actual Golden", fmt["label"]); ws.write_number("G13", int(y_true.sum()), fmt["value"])
    ws.write("F14", "Actual NOT Golden", fmt["label"]); ws.write_number("G14", int((y_true == 0).sum()), fmt["value"])
    ws.write("F15", "Pred Golden", fmt["label"]); ws.write_number("G15", int(y_pred.sum()), fmt["value"])
    ws.write("F16", "Pred NOT Golden", fmt["label"]); ws.write_number("G16", int((y_pred == 0).sum()), fmt["value"])

    # 5. Feature importance
    ws = wb.add_worksheet("Feature importance")
    ws.hide_gridlines(2)
    ws.set_zoom(100)
    ws.set_column("A:A", 3)
    ws.set_column("B:B", 38)
    ws.set_column("C:C", 18)
    ws.set_column("D:D", 18)
    _write_df(ws, importance, 1, 1, fmt, percent_cols=["Importance_%"])
    if len(importance) and importance["Importance"].notna().any():
        last = 2 + len(importance)
        ws.conditional_format(f"C3:C{last}", {"type": "data_bar", "bar_color": "#5B9BD5", "bar_solid": True})
        ws.conditional_format(f"D3:D{last}", {"type": "data_bar", "bar_color": "#70AD47", "bar_solid": True})
    ws.freeze_panes(2, 1)
    ws.autofilter(1, 1, 1 + len(importance), 3)

    # 6. Feature describe
    ws = wb.add_worksheet("Feature describe")
    ws.hide_gridlines(2)
    ws.set_zoom(100)
    ws.set_column("A:A", 3)
    ws.set_column("B:B", 38)
    ws.set_column("C:C", 18)
    ws.set_column("D:D", 70)
    _write_df(ws, feature_desc, 1, 1, fmt)
    ws.freeze_panes(2, 1)
    ws.autofilter(1, 1, 1 + len(feature_desc), 3)

    wb.close()
    return output_path


if __name__ == "__main__":
    print("Import create_golden_model_report() and pass your model objects/data.")
