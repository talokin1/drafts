def transform_date(value):
    if pd.isna(value):
        return None

    parts = [part.strip() for part in value.split(',')]
    if len(parts) == 2:
        month, year = parts
        month_num = month_map.get(month)
        if month_num:
            return f"{year}_{month_num}"

    return None

df["year_month"] = df["Дата дзвінка"].apply(transform_date)

for ym, g in df.dropna(subset=["year_month"]).groupby("year_month"):
    filename = f"NPS_{ym}.parquet"
    g.to_parquet(os.path.join(FILEPATH, filename), index=False)
    print(f"Saved {filename}. Rows = {len(g)}")



total = len(df)
parsed = df["year_month"].notna().sum()
missing = df["year_month"].isna().sum()

print(f"Total rows:   {total}")
print(f"Parsed rows:  {parsed}")
print(f"Missing rows:{missing}")
print(f"Parsed %:    {parsed / total:.2%}")
