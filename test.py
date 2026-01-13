df_acq = df_batch[df_batch["is_acquiring"]].copy()

if df_acq.empty:
    print("No acquiring in batch")


agg_batch = (
    df_acq
    .groupby("CONTRAGENTBIDENTIFYCODE")
    .agg(
        SUMMAEQ=("SUMMAEQ", "sum"),
        NUM_TRX=("SUMMAEQ", "size"),
        PLATPURPOSE=("PLATPURPOSE", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
        CONTRAGENTASNAME=("CONTRAGENTASNAME", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
    )
    .reset_index()
    .rename(columns={"CONTRAGENTBIDENTIFYCODE": "IDENTIFYCODE"})
)

agg_batch["IDENTIFYCODE"] = agg_batch["IDENTIFYCODE"].astype(str)
agg_batch["MONTH"] = "2025-11"


all_batches = pd.concat(
    pd.read_parquet(f)
    for f in Path("C:/Projects/(DS-398) Acquiring/all_corr_2025_11").glob("acq_2025-11-*.parquet")
)

df_acq = all_batches[all_batches["is_acquiring"]]

final_agg = (
    df_acq
    .groupby("CONTRAGENTBIDENTIFYCODE")
    .agg(
        SUMMAEQ=("SUMMAEQ", "sum"),
        NUM_TRX=("SUMMAEQ", "size"),
        PLATPURPOSE=("PLATPURPOSE", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
        CONTRAGENTASNAME=("CONTRAGENTASNAME", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
    )
    .reset_index()
    .rename(columns={"CONTRAGENTBIDENTIFYCODE": "IDENTIFYCODE"})
)

final_agg["MONTH"] = "2025-11"
