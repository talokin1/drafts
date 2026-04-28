import pandas as pd
import os

income_months = [
    '2025_04','2025_05','2025_06','2025_07','2025_08',
    '2025_09','2025_10','2025_11','2025_12',
    '2026_01','2026_02'
]

base_path = r"M:\Controlling\Data_Science_Projects\Income_Data"

target_contragent_id = 3044909  # <-- підстав свого клієнта

results = []

for month in income_months:
    file_path = os.path.join(
        base_path,
        f"income_wide_corporate_clients_{month}.csv"
    )
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # фільтр клієнта
    client_row = df[df['CONTRAGENTID'] == target_contragent_id]

    if client_row.empty:
        income = 0
    else:
        # якщо раптом декілька рядків — беремо суму
        income = client_row['MONTHLY_INCOME'].sum()

    results.append({
        'month': month,
        'income': income
    })

# фінальний датафрейм
result_df = pd.DataFrame(results)

# сумарний інком
total_income = result_df['income'].sum()

result_df['cumulative_income'] = result_df['income'].cumsum()

print(result_df)
print(f"\nTOTAL INCOME: {total_income}")