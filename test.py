from golden_model_report import create_golden_model_report

create_golden_model_report(
    y_true=y,
    oof_proba=oof_proba,
    inference_df=result_df,
    model=models,
    X_reference=X,
    threshold=BEST_THRESHOLD,
    output_path="Golden_Model_Report.xlsx"
)