import pandas as pd
import numpy as np
import gc

files = [
    r'M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_10.parquet',
    r'M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_11.parquet',
    r'M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2025_12.parquet',
    r'M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2026_01.parquet',
    r'M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2026_02.parquet',
    r'M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw\data_trxs_2026_03.parquet'
]

chunk_list = []

for file in files:
    print(f"Обробка файлу: {file.split('Raw')[-1]}...")
    
    temp_df = pd.read_parquet(file)
    
    temp_df = temp_df[temp_df["BANKBID"].astype(str).str.startswith("8")]
    
    temp_df["CONTRAGENTAIDENTIFYCODE"] = temp_df["CONTRAGENTAIDENTIFYCODE"].astype(str)
    temp_df["CONTRAGENTBIDENTIFYCODE"] = temp_df["CONTRAGENTBIDENTIFYCODE"].astype(str)
    
    temp_df = temp_df[
        temp_df["CONTRAGENTAIDENTIFYCODE"].isin(clients_ids) | 
        temp_df["CONTRAGENTBIDENTIFYCODE"].isin(clients_ids)
    ]
    
    chunk_list.append(temp_df)
    
    del temp_df
    gc.collect()

print("Конкатенація результатів...")
client_trx = pd.concat(chunk_list, ignore_index=True)

print(f"Готово! Загальна кількість транзакцій за півроку: {len(client_trx)}")
