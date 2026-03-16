import pandas as pd
import os

# мапа місяців
month_map = {
    1: "Січень",
    2: "Лютий",
    3: "Березень",
    4: "Квітень",
    5: "Травень",
    6: "Червень",
    7: "Липень",
    8: "Серпень",
    9: "Вересень",
    10: "Жовтень",
    11: "Листопад",
    12: "Грудень"
}

# гарантуємо datetime
df["call_date"] = pd.to_datetime(df["call_date"])

# створюємо назву місяця
df["year_month"] = df["call_date"].apply(
    lambda x: f"{month_map[x.month]}_{x.year}"
)

sum_rows = 0

for ym, g in df.groupby("year_month"):

    filename = f"NPS_{ym}.parquet"

    g.to_parquet(
        os.path.join(FILEPATH, filename),
        index=False
    )

    sum_rows += len(g)

    print(f"Saved {filename}. Rows = {len(g)}")

total = len(df)
parsed = df["year_month"].notna().sum()
missing = df["year_month"].isna().sum()

print("Total rows:", total)
print("Parsed rows:", parsed)
print("Missing rows:", missing)
print("Parsed %:", parsed / total)
print("Total saved:", sum_rows)