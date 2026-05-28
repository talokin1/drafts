# деталізація транзакцій по конвертованих клієнтах
converted_fx = fx_by_may[
    fx_by_may["CONTRAGENTID"].astype(str).str.strip().isin(converted_ids)
].copy()

converted_fx.head()


# сума FX по клієнтах
conversion_summary = (
    converted_fx
    .groupby("CONTRAGENTID", as_index=False)
    .agg(
        fx_operations=("CONTRAGENTID", "count"),
        fx_total_amount=("AMOUNT", "sum")
    )
)

conversion_summary




converted_fx["ARCDATE"] = pd.to_datetime(converted_fx["ARCDATE"])

fx_grouped = (
    converted_fx
    .groupby("CONTRAGENTID", as_index=False)
    .agg(
        first_fx_date=("ARCDATE", "min"),
        last_fx_date=("ARCDATE", "max"),
        fx_days=("ARCDATE", lambda x: x.dt.date.nunique()),
        fx_operations=("ARCDATE", "count"),
        fx_total_amount=("AMOUNT", "sum")
    )
    .sort_values("first_fx_date")
)

fx_grouped



fx_dates_by_client = (
    converted_fx
    .groupby("CONTRAGENTID", as_index=False)
    .agg(
        fx_dates=("ARCDATE", lambda x: " | ".join(sorted(x.dt.strftime("%Y-%m-%d").unique()))),
        fx_operations=("ARCDATE", "count")
    )
)

fx_dates_by_client
