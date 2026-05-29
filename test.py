def evaluate_thresholds(y_true, proba, hnwi_threshold, prem_threshold):
    preds = []

    for _, row in proba.iterrows():
        if row["P_HNWI"] >= hnwi_threshold:
            preds.append("HNWI")
        elif row["P_PREM"] >= prem_threshold:
            preds.append("PREM")
        else:
            preds.append("MASS")

    preds = pd.Series(preds, index=y_true.index)

    report = precision_recall_fscore_support(
        y_true,
        preds,
        labels=["MASS", "PREM", "HNWI"],
        zero_division=0
    )

    precision, recall, f1, support = report

    result = {
        "hnwi_threshold": hnwi_threshold,
        "prem_threshold": prem_threshold,

        "mass_precision": precision[0],
        "prem_precision": precision[1],
        "hnwi_precision": precision[2],

        "mass_recall": recall[0],
        "prem_recall": recall[1],
        "hnwi_recall": recall[2],

        "mass_f1": f1[0],
        "prem_f1": f1[1],
        "hnwi_f1": f1[2],

        "macro_f1": f1_score(y_true, preds, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),

        "n_pred_mass": (preds == "MASS").sum(),
        "n_pred_prem": (preds == "PREM").sum(),
        "n_pred_hnwi": (preds == "HNWI").sum()
    }

    # Бізнес-орієнтована функція якості.
    # HNWI важливіший, PREM другий, MASS третій.
    result["business_score"] = (
        0.45 * result["hnwi_recall"] +
        0.25 * result["prem_recall"] +
        0.15 * result["mass_recall"] +
        0.15 * result["macro_f1"]
    )

    return result


threshold_results = []

hnwi_grid = np.arange(0.05, 0.61, 0.025)
prem_grid = np.arange(0.20, 0.81, 0.025)

for hnwi_thr in hnwi_grid:
    for prem_thr in prem_grid:
        res = evaluate_thresholds(
            y_true=y_valid,
            proba=valid_proba,
            hnwi_threshold=hnwi_thr,
            prem_threshold=prem_thr
        )
        threshold_results.append(res)

threshold_results_df = pd.DataFrame(threshold_results)

threshold_results_df = threshold_results_df.sort_values(
    by="business_score",
    ascending=False
)

display(threshold_results_df.head(20))





best_row = threshold_results_df.iloc[0]

best_hnwi_threshold = best_row["hnwi_threshold"]
best_prem_threshold = best_row["prem_threshold"]

print("Best HNWI threshold:", best_hnwi_threshold)
print("Best PREM threshold:", best_prem_threshold)
print(best_row)







y_pred_threshold = model.predict(
    X_valid,
    hnwi_threshold=best_hnwi_threshold,
    prem_threshold=best_prem_threshold,
    mode="threshold"
)

print("Classification report — threshold mode:")
print(classification_report(
    y_valid,
    y_pred_threshold,
    labels=["MASS", "PREM", "HNWI"]
))

print("Confusion matrix — threshold mode:")
cm = confusion_matrix(
    y_valid,
    y_pred_threshold,
    labels=["MASS", "PREM", "HNWI"]
)

cm_df = pd.DataFrame(
    cm,
    index=["true_MASS", "true_PREM", "true_HNWI"],
    columns=["pred_MASS", "pred_PREM", "pred_HNWI"]
)

display(cm_df)

print("Balanced accuracy:", balanced_accuracy_score(y_valid, y_pred_threshold))
print("Macro F1:", f1_score(y_valid, y_pred_threshold, average="macro"))














def top_k_hnwi_analysis(y_true, proba, k_values=[25, 50, 100, 150, 200]):
    tmp = pd.DataFrame({
        "true_segment": y_true,
        "P_HNWI": proba["P_HNWI"],
        "P_PREM": proba["P_PREM"],
        "P_MASS": proba["P_MASS"]
    })

    tmp = tmp.sort_values("P_HNWI", ascending=False)

    total_hnwi = (tmp["true_segment"] == "HNWI").sum()

    rows = []

    for k in k_values:
        top_k = tmp.head(k)

        found_hnwi = (top_k["true_segment"] == "HNWI").sum()
        found_prem = (top_k["true_segment"] == "PREM").sum()
        found_mass = (top_k["true_segment"] == "MASS").sum()

        rows.append({
            "top_k": k,
            "found_hnwi": found_hnwi,
            "found_prem": found_prem,
            "found_mass": found_mass,
            "hnwi_recall_at_k": found_hnwi / total_hnwi if total_hnwi > 0 else np.nan,
            "hnwi_precision_at_k": found_hnwi / k
        })

    return pd.DataFrame(rows)


topk_df = top_k_hnwi_analysis(
    y_true=y_valid,
    proba=valid_proba,
    k_values=[10, 25, 50, 100, 150, 200]
)

display(topk_df)










valid_result = X_valid.copy()

valid_result["true_segment"] = y_valid
valid_result["P_MASS"] = valid_proba["P_MASS"]
valid_result["P_PREM"] = valid_proba["P_PREM"]
valid_result["P_HNWI"] = valid_proba["P_HNWI"]

valid_result["predicted_segment"] = y_pred_threshold

def make_business_group(row):
    if row["P_HNWI"] >= 0.50:
        return "Strong HNWI candidate"
    elif row["P_HNWI"] >= best_hnwi_threshold:
        return "Weak HNWI candidate"
    elif row["P_PREM"] >= best_prem_threshold:
        return "PREM candidate"
    else:
        return "Likely MASS"

valid_result["business_group"] = valid_result.apply(make_business_group, axis=1)

valid_result_sorted = valid_result.sort_values("P_HNWI", ascending=False)

display(valid_result_sorted.head(50))













final_model = HierarchicalSegmentModel(
    cat_cols=cat_cols,
    stage1_positive_boost=1.3,
    stage2_hnwi_boost=2.5,
    random_state=42
)

final_model.fit(
    X,
    y
)



X_external = df_external[feature_cols].copy()

external_proba = final_model.predict_proba(X_external)

external_pred = final_model.predict(
    X_external,
    hnwi_threshold=best_hnwi_threshold,
    prem_threshold=best_prem_threshold,
    mode="threshold"
)

external_result = df_external.copy()

external_result["P_MASS"] = external_proba["P_MASS"].values
external_result["P_PREM"] = external_proba["P_PREM"].values
external_result["P_HNWI"] = external_proba["P_HNWI"].values

external_result["predicted_segment"] = external_pred.values

external_result["business_group"] = external_result.apply(
    lambda row: (
        "Strong HNWI candidate" if row["P_HNWI"] >= 0.50 else
        "Weak HNWI candidate" if row["P_HNWI"] >= best_hnwi_threshold else
        "PREM candidate" if row["P_PREM"] >= best_prem_threshold else
        "Likely MASS"
    ),
    axis=1
)

external_result = external_result.sort_values("P_HNWI", ascending=False)

display(external_result.head(100))







import joblib

artifacts = {
    "model": final_model,
    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "best_hnwi_threshold": best_hnwi_threshold,
    "best_prem_threshold": best_prem_threshold
}

joblib.dump(artifacts, "hierarchical_hnwi_model.pkl")