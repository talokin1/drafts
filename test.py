import numpy as np
import pandas as pd

# основний датасет
result = fin_ind[["IDENTIFYCODE", "FIRM_TELORG"]].copy()

# новіші телефони
phones = pd.read_csv(
    r"C:\Projects\YouControl data\(DS-772) Get phone data\phone_results.csv",
    dtype={"edrpou": "str"}
).rename(columns={"edrpou": "IDENTIFYCODE"})

# приводимо ключі до одного формату
result["IDENTIFYCODE"] = result["IDENTIFYCODE"].astype(str).str.strip()
phones["IDENTIFYCODE"] = phones["IDENTIFYCODE"].astype(str).str.strip()

# залишаємо тільки потрібні колонки
phones = phones[["IDENTIFYCODE", "phones_clean"]].copy()

# якщо в phones є дублікати по IDENTIFYCODE, об'єднуємо телефони
phones = (
    phones
    .dropna(subset=["phones_clean"])
    .groupby("IDENTIFYCODE", as_index=False)["phones_clean"]
    .agg(lambda x: "; ".join(pd.unique(x.astype(str))))
)

# merge
merged = result.merge(
    phones,
    on="IDENTIFYCODE",
    how="left"
)

# фінальний телефон:
# 1) якщо є новіші phones_clean — беремо їх
# 2) якщо немає — залишаємо старий FIRM_TELORG
# 3) якщо немає нічого — no phone
merged["PHONE_FINAL"] = np.where(
    merged["phones_clean"].notna() & (merged["phones_clean"].astype(str).str.strip() != ""),
    merged["phones_clean"],
    merged["FIRM_TELORG"]
)

merged["PHONE_FINAL"] = (
    merged["PHONE_FINAL"]
    .fillna("no phone")
    .replace("", "no phone")
)

# фінальний результат
final_result = merged[["IDENTIFYCODE", "PHONE_FINAL"]].copy()

final_result

final_result = merged[
    ["IDENTIFYCODE", "FIRM_TELORG", "phones_clean", "PHONE_FINAL"]
].copy()