import pandas as pd
import os
from pathlib import Path

INPUT_PATH = r"C:\Projects\(DS-514) NPS Preprocessing\NPS_RAW"
OUTPUT_PATH = r"C:\Projects\(DS-514) NPS Preprocessing\PARQUET"

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

files = Path(INPUT_PATH).glob("*.xlsx")

for file in files:

    print(f"Processing {file.name}")

    df = pd.read_excel(file, dtype={"CNUM": "str"})

    df["call_date"] = pd.to_datetime(df["call_date"])

    df["year_month"] = (
        df["call_date"]
        .dt.month.map(month_map)
        + "_"
        + df["call_date"].dt.year.astype(str)
    )

    for ym, g in df.groupby("year_month"):

        filename = f"NPS_{ym}.parquet"

        g.to_parquet(
            os.path.join(OUTPUT_PATH, filename),
            index=False
        )

        print(f"Saved {filename} | rows={len(g)}")