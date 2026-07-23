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

clients_excel = clients.rename(columns={
    "MOBILEPHONE": "Номер телефону",
    "hnwi_probability": "Ймовірність до HNWI",
    "cars_count": "Кількість авто",
    "unique_marks": "Унікальні марки",
    "max_price_usd": "Максимальна ціна",
    "median_price_usd": "Медіанна ціна авто",
    "main_car_mark": "Основне авто",
    "main_car_category": "Категорія авто",
    "main_car_city": "Регіон",
    "top_3_cars": "Топ 3 авто"
})[["Номер телефону", "Ймовірність до HNWI", "Кількість авто", "Унікальні марки", "Максимальна ціна", "Медіанна ціна авто", "Основне авто", "Категорія авто", "Регіон", "Топ 3 авто"]]

cars_excel = cars.rename(columns={
    "MOBILEPHONE": "Номер телефону",
    "mark_name": "Назва марки",
    "category_name": "Категорія авто",
    "price_usd": "Ціна ($)",
    "location_city": "Регіон"
})[["Номер телефону", "Назва марки", "Категорія авто", "Ціна ($)", "Регіон"]]




with pd.ExcelWriter("HNWI_candidates.xlsx", engine="xlsxwriter") as writer:
    clients_excel.to_excel(writer, sheet_name="HNWI клієнти", index=False)
    cars_excel.to_excel(writer, sheet_name="Автомобілі", index=False)

    ws_clients = writer.sheets["HNWI клієнти"]
    ws_cars = writer.sheets["Автомобілі"]
    percent_format = writer.book.add_format({"num_format": "0.00%"})

    ws_clients.freeze_panes(1, 0)
    ws_clients.autofilter(0, 0, len(clients_excel), len(clients_excel.columns) - 1)
    ws_clients.set_column("A:A", 17)
    ws_clients.set_column("B:B", 22, percent_format)
    ws_clients.set_column("C:F", 18)
    ws_clients.set_column("G:I", 22)
    ws_clients.set_column("J:J", 80)

    ws_cars.freeze_panes(1, 0)
    ws_cars.autofilter(0, 0, len(cars_excel), len(cars_excel.columns) - 1)
    ws_cars.set_column("A:A", 17)
    ws_cars.set_column("B:C", 25)
    ws_cars.set_column("D:D", 15)
    ws_cars.set_column("E:E", 20)