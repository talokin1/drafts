test_clients_agg = (
    test_clients
    .groupby("IDENTIFYCODE", as_index=False)
    .agg({
        "CONTRAGENTID": lambda x: ",".join(sorted(x.dropna().astype(str).unique())),
        "INCOME_LIABILITIES": "sum",
        "INCOME_ASSETS": "sum",
        "COMMISSIONS": "sum",
        "MONTHLY_INCOME": "sum"
    })
)


print("test_clients rows before:", len(test_clients))
print("test_clients rows after agg:", len(test_clients_agg))
print("duplicated IDENTIFYCODE after agg:", test_clients_agg["IDENTIFYCODE"].duplicated().sum())


df_tmp = df.merge(
    test_clients_agg,
    how="left",
    on="IDENTIFYCODE",
    validate="m:1"
)

print("df rows before merge:", len(df))
print("df rows after merge:", len(df_tmp))












# -----------------------------
# 1. Нормалізація ключів
# -----------------------------

def normalize_id(s):
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .str.zfill(8)
    )

active_clients = active_clients.copy()
income_data = income_data.copy()
df = df.copy()

active_clients["IDENTIFYCODE"] = normalize_id(active_clients["IDENTIFYCODE"])
df["IDENTIFYCODE"] = normalize_id(df["IDENTIFYCODE"])

active_clients["CONTRAGENTID"] = active_clients["CONTRAGENTID"].astype(str).str.strip()
income_data["CONTRAGENTID"] = income_data["CONTRAGENTID"].astype(str).str.strip()


# -----------------------------
# 2. Спочатку агрегуємо income_data на рівні CONTRAGENTID
# -----------------------------

income_by_contragent = (
    income_data
    .groupby("CONTRAGENTID", as_index=False)
    .agg({
        "INCOME_LIABILITIES": "sum",
        "INCOME_ASSETS": "sum",
        "COMMISSIONS": "sum",
        "MONTHLY_INCOME": "sum"
    })
)


# -----------------------------
# 3. Мержимо active_clients з income
# -----------------------------

test_clients = active_clients.merge(
    income_by_contragent,
    how="left",
    on="CONTRAGENTID",
    validate="m:1"
)


# -----------------------------
# 4. Агрегуємо до одного рядка на IDENTIFYCODE
# -----------------------------

test_clients_agg = (
    test_clients
    .groupby("IDENTIFYCODE", as_index=False)
    .agg({
        "CONTRAGENTID": lambda x: ",".join(sorted(x.dropna().astype(str).unique())),
        "INCOME_LIABILITIES": "sum",
        "INCOME_ASSETS": "sum",
        "COMMISSIONS": "sum",
        "MONTHLY_INCOME": "sum"
    })
)


# -----------------------------
# 5. Безпечний merge у df
# -----------------------------

df_check = df.merge(
    test_clients_agg,
    how="left",
    on="IDENTIFYCODE",
    validate="m:1"
)

print("df rows:", len(df))
print("df_check rows:", len(df_check))
print("duplicated IDENTIFYCODE in right table:", test_clients_agg["IDENTIFYCODE"].duplicated().sum())



