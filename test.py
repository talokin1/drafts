client_summary = (
    result
    .groupby("CONTRAGENTID", as_index=False)
    .agg(
        is_acquiring=("is_acquiring", "max")
    )
)

client_counts = (
    client_summary["is_acquiring"]
    .value_counts(dropna=False)
    .rename_axis("is_acquiring")
    .reset_index(name="clients_count")
)

client_counts