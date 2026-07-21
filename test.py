client_df = (
    df.groupby("CONTRAGENTID", as_index=False)
      .agg(
          group_id=("group_id", "first"),
          SEGMENT=("SEGMENT", "first"),

          cars_count=("price_usd", "size"),
          categories_count=("category_name", "nunique"),

          price_max=("price_usd", "max"),
          price_mean=("price_usd", "mean"),
          price_median=("price_usd", "median"),
          price_min=("price_usd", "min"),
          price_std=("price_usd", "std"),

          mileage_min=("mileage", "min"),
          mileage_mean=("mileage", "mean"),
          mileage_median=("mileage", "median"),

          doors_max=("doors_count", "max"),
          doors_median=("doors_count", "median"),

          has_report_share=("has_report", "mean"),
          is_abroad_any=("is_abroad", "max"),
          exchange_possible_any=("exchange_possible", "max"),
          auction_possible_any=("auction_possible", "max"),

          premium_mark_share=("premium_mark", "mean")
      )
      .merge(top_car, on="CONTRAGENTID", how="left")
)

print(client_df.shape)
print(client_df["SEGMENT"].value_counts())