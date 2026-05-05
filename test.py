df_check["over_error"] = df_check["y_pred"] - df_check["y_true"]

df_check.sort_values("over_error", ascending=False).head(50)


for seg in df_check["segment"].unique():
    print("=" * 80)
    print(seg)
    display(
        df_check[df_check["segment"] == seg]
        .sort_values("over_error", ascending=False)
        .head(20)
    )




nonzero_pred = df_check[df_check["y_pred"] > 0]

nonzero_pred.groupby("segment")["y_pred"].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
)


nonzero_true = df_check[df_check["y_true"] > 0]

nonzero_true.groupby("segment")["y_true"].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
)

