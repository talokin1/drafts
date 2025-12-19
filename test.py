import pandas as pd
from pathlib import Path

files = [
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_02.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_03.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_04.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_05.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_06.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_07.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_08.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_09.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_10.parquet",
    r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_11.parquet",
]

output_path = r"M:\Controlling\Data_Science_Projects\Corp_Churn\Results\2924_flows.xlsx"

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    for file in files:
        df = pd.read_parquet(file)

        df["ACCOUNTANO"] = df["ACCOUNTANO"].astype(str)
        df["ACCOUNTBNO"] = df["ACCOUNTBNO"].astype(str)

        mask = (
            (df["BANKAID"] != 300528) &
            (df["ACCOUNTANO"].str.startswith("2924")) &
            (
                df["ACCOUNTBNO"].str.startswith("2600") |
                df["ACCOUNTBNO"].str.startswith("2650") |
                df["ACCOUNTBNO"].str.startswith("2655")
            )
        )

        df_filtered = df[mask]

        month = Path(file).stem.replace("data_trxs_", "")  # 2025_02
        sheet_name = month.replace("_", "-")               # 2025-02

        df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)
