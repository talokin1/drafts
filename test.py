import pandas as pd
from pathlib import Path

trxs = pd.read_parquet(
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_11.parquet",
    columns=[  # ОБОВʼЯЗКОВО
        "ARCDATE",
        "PLATPURPOSE",
        "CONTRAGENTASNAME",
        # + тільки те, що реально потрібно
    ]
)

trxs["ARCDATE"] = pd.to_datetime(trxs["ARCDATE"]).dt.date



dates = sorted(trxs["ARCDATE"].unique())

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


OUT_DIR = Path("M:/Controlling/tmp/acquiring_batches")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def process_batch(df: pd.DataFrame) -> pd.DataFrame:
    # тут твоя логіка regex / score / flags
    df["is_acquiring"] = df["PLATPURPOSE"].apply(detect_acquiring)
    return df[["ARCDATE", "is_acquiring"]]



for batch_dates in chunks(dates, 10):
    batch_key = f"{batch_dates[0]}_{batch_dates[-1]}"
    out_file = OUT_DIR / f"acq_{batch_key}.parquet"

    if out_file.exists():
        print(f"skip {batch_key}")
        continue

    batch_df = trxs[trxs["ARCDATE"].isin(batch_dates)]
    result = process_batch(batch_df)

    result.to_parquet(out_file, index=False)
    print(f"saved {out_file}")


files = list(OUT_DIR.glob("acq_*.parquet"))
final = pd.concat(
    (pd.read_parquet(f) for f in files),
    ignore_index=True
)

final.to_parquet("acquiring_final.parquet", index=False)



