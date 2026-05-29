model = HierarchicalSegmentModel(
    cat_cols=cat_cols,
    stage1_positive_boost=1.2,
    stage2_hnwi_boost=1.3,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    X_valid=X_valid,
    y_valid=y_valid
)

valid_proba = model.predict_proba(X_valid)





threshold_results = []

hnwi_grid = np.arange(0.10, 0.81, 0.025)
prem_grid = np.arange(0.20, 0.91, 0.025)

for hnwi_thr in hnwi_grid:
    for prem_thr in prem_grid:
        res = evaluate_thresholds(
            y_true=y_valid,
            proba=valid_proba,
            hnwi_threshold=hnwi_thr,
            prem_threshold=prem_thr
        )

        res["business_score_balanced"] = (
            0.25 * res["hnwi_recall"] +
            0.20 * res["hnwi_precision"] +
            0.20 * res["prem_recall"] +
            0.15 * res["mass_recall"] +
            0.20 * res["macro_f1"]
        )

        res["business_score_discovery"] = (
            0.45 * res["hnwi_recall"] +
            0.25 * res["hnwi_precision"] +
            0.15 * res["prem_recall"] +
            0.10 * res["mass_recall"] +
            0.05 * res["macro_f1"]
        )

        threshold_results.append(res)

threshold_results_df = pd.DataFrame(threshold_results)
















balanced_candidates = threshold_results_df[
    (threshold_results_df["hnwi_precision"] >= 0.12) &
    (threshold_results_df["hnwi_recall"] >= 0.50) &
    (threshold_results_df["mass_recall"] >= 0.70) &
    (threshold_results_df["prem_recall"] >= 0.20)
].copy()

display(
    balanced_candidates
    .sort_values("business_score_balanced", ascending=False)
    .head(20)
)







discovery_candidates = threshold_results_df[
    (threshold_results_df["hnwi_recall"] >= 0.70) &
    (threshold_results_df["hnwi_precision"] >= 0.10) &
    (threshold_results_df["mass_recall"] >= 0.55)
].copy()

display(
    discovery_candidates
    .sort_values("business_score_discovery", ascending=False)
    .head(20)
)