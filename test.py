month_map = {
    "Січень": "01", "Лютий": "02", "Березень": "03", "Квітень": "04",
    "Травень": "05", "Червень": "06", "Липень": "07", "Серпень": "08",
    "Вересень": "09", "Жовтень": "10", "Листопад": "11", "Грудень": "12"
}

tmp = df["Дата дзвінка"].str.split(", ", expand=True)

df["month_name"] = tmp[0]
df["year"] = tmp[1]
df["month_num"] = df["month_name"].map(month_map)

df["year_month"] = df["year"] + "_" + df["month_num"]

for ym, g in df.groupby("year_month"):
    filename = f"NPS_{ym}.parquet"
    g.to_parquet(filename, index=False)
    print(f"Saved {filename} | Rows = {len(g)}")

df.groupby("year_month").size().sum() == len(df)
