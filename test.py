car_cols = ["MOBILEPHONE", "mark_name", "category_name", "price_usd", "location_city"]
cars = full_df[car_cols].drop_duplicates()
cars = cars[cars["MOBILEPHONE"].isin(temp["MOBILEPHONE"])].sort_values(["MOBILEPHONE", "price_usd"], ascending=[True, False])

stats = cars.groupby("MOBILEPHONE").agg(cars_count=("price_usd", "size"), unique_marks=("mark_name", "nunique"), max_price_usd=("price_usd", "max"), median_price_usd=("price_usd", "median")).reset_index()

main_car = cars.drop_duplicates("MOBILEPHONE").rename(columns={"mark_name": "main_car_mark", "category_name": "main_car_category", "price_usd": "main_car_price_usd", "location_city": "main_car_city"})

top3 = cars.groupby("MOBILEPHONE").head(3).copy()
top3["car_info"] = top3["mark_name"].fillna("") + " | " + top3["category_name"].fillna("") + " | $" + top3["price_usd"].round().astype("Int64").astype(str)
top3 = top3.groupby("MOBILEPHONE")["car_info"].agg("; ".join).rename("top_3_cars").reset_index()

clients = temp[["MOBILEPHONE", "NAME", "hnwi_probability", "is_hnwi"]].drop_duplicates("MOBILEPHONE")
clients = clients.merge(stats, on="MOBILEPHONE", how="left").merge(main_car, on="MOBILEPHONE", how="left").merge(top3, on="MOBILEPHONE", how="left")
clients["is_probable_dealer"] = (clients["cars_count"] >= clients["cars_count"].quantile(0.95)).astype(int)
clients["priority_rank"] = clients["hnwi_probability"].rank(method="first", ascending=False).astype(int)
clients = clients.sort_values(["is_probable_dealer", "hnwi_probability", "max_price_usd"], ascending=[True, False, False])




with pd.ExcelWriter("HNWI_candidates.xlsx", engine="xlsxwriter") as writer:
    clients.to_excel(writer, sheet_name="HNWI_clients", index=False)
    cars.to_excel(writer, sheet_name="Cars_detail", index=False)
    for sheet, df in {"HNWI_clients": clients, "Cars_detail": cars}.items():
        ws = writer.sheets[sheet]
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)