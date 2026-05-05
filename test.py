df_check = pd.DataFrame({
    "y_true": y_inference_true,
    "y_pred": y_inference_pred,
    "segment": inference_df["FIRM_TYPE"]
})

df_check["abs_error"] = np.abs(df_check["y_true"] - df_check["y_pred"])
df_check["ratio"] = df_check["y_pred"] / df_check["y_true"].replace(0, np.nan)

summary = df_check.groupby("segment").agg(
    count=("y_true", "size"),
    true_sum=("y_true", "sum"),
    pred_sum=("y_pred", "sum"),
    true_mean=("y_true", "mean"),
    pred_mean=("y_pred", "mean"),
    mae=("abs_error", "mean"),
    median_true=("y_true", "median"),
    median_pred=("y_pred", "median"),
    zero_true_rate=("y_true", lambda x: (x == 0).mean()),
    zero_pred_rate=("y_pred", lambda x: (x == 0).mean()),
)

summary["sum_ratio"] = summary["pred_sum"] / summary["true_sum"]

summary