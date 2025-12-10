purpose_by_type = (
    merged.groupby(["CLIENT_IDENTIFYCODE", "TYPE"])["PLATPURPOSE"]
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
    .reset_index()
)

purpose_wide = (
    purpose_by_type
    .pivot(index="CLIENT_IDENTIFYCODE", columns="TYPE", values="PLATPURPOSE")
    .rename(columns={
        "DEBIT_SELF_ACQ": "PLATPURPOSE_DEBIT",
        "CREDIT_SELF_ACQ": "PLATPURPOSE_CREDIT"
    })
    .reset_index()
)

summary = summary.merge(purpose_wide, on="CLIENT_IDENTIFYCODE", how="left")

summary["PLATPURPOSE"] = summary.apply(
    lambda row: row["PLATPURPOSE_DEBIT"] if row["TYPE"] == "DEBIT_SELF_ACQ"
                else row["PLATPURPOSE_CREDIT"],
    axis=1
)

summary = summary.drop(columns=["PLATPURPOSE_DEBIT", "PLATPURPOSE_CREDIT"])
