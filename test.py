import os
import json
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

PATH = r"M:\Controlling\Data_Science_Projects\Corp_Churn\Data\Raw"

MFO_OTP = 300528

transaction_files = [
    os.path.join(PATH, file)
    for file in os.listdir(PATH)
    if "data_trxs_2026_" in file and file.endswith(".parquet")
]

# Якщо треба додати ще 2025_11, 2025_12 тощо:
# transaction_files += [
#     os.path.join(PATH, file)
#     for file in os.listdir(PATH)
#     if "data_trxs_2025_" in file and file.endswith(".parquet")
# ]



sample = sample.copy()

sample["IDENTIFYCODE"] = (
    sample["IDENTIFYCODE"]
    .astype(str)
    .str.strip()
)

sample_codes = set(sample["IDENTIFYCODE"])



use_cols = [
    "CONTRAGENTAIDENTIFYCODE",
    "CONTRAGENTBIDENTIFYCODE",
    "BANKAID",
    "BANKBID",
]


# Структура:
# bank_usage["12345678"][300528] = 15
# bank_usage["12345678"][305299] = 3

bank_usage = defaultdict(Counter)

bank_usage_sender = defaultdict(Counter)
bank_usage_receiver = defaultdict(Counter)

for file in tqdm(transaction_files, desc="Processing transaction files"):
    trx = pd.read_parquet(file, columns=use_cols)

    trx["CONTRAGENTAIDENTIFYCODE"] = (
        trx["CONTRAGENTAIDENTIFYCODE"]
        .astype(str)
        .str.strip()
    )

    trx["CONTRAGENTBIDENTIFYCODE"] = (
        trx["CONTRAGENTBIDENTIFYCODE"]
        .astype(str)
        .str.strip()
    )

    # -------------------------
    # Клієнт як відправник
    # -------------------------
    sender_part = trx[
        trx["CONTRAGENTAIDENTIFYCODE"].isin(sample_codes)
    ][["CONTRAGENTAIDENTIFYCODE", "BANKAID"]].copy()

    sender_part = sender_part.dropna(subset=["BANKAID"])

    sender_part["BANKAID"] = sender_part["BANKAID"].astype("Int64")

    sender_counts = (
        sender_part
        .groupby(["CONTRAGENTAIDENTIFYCODE", "BANKAID"])
        .size()
        .reset_index(name="txn_cnt")
    )

    for row in sender_counts.itertuples(index=False):
        client_code = row.CONTRAGENTAIDENTIFYCODE
        bank_id = int(row.BANKAID)
        cnt = int(row.txn_cnt)

        bank_usage[client_code][bank_id] += cnt
        bank_usage_sender[client_code][bank_id] += cnt

    # -------------------------
    # Клієнт як отримувач
    # -------------------------
    receiver_part = trx[
        trx["CONTRAGENTBIDENTIFYCODE"].isin(sample_codes)
    ][["CONTRAGENTBIDENTIFYCODE", "BANKBID"]].copy()

    receiver_part = receiver_part.dropna(subset=["BANKBID"])

    receiver_part["BANKBID"] = receiver_part["BANKBID"].astype("Int64")

    receiver_counts = (
        receiver_part
        .groupby(["CONTRAGENTBIDENTIFYCODE", "BANKBID"])
        .size()
        .reset_index(name="txn_cnt")
    )

    for row in receiver_counts.itertuples(index=False):
        client_code = row.CONTRAGENTBIDENTIFYCODE
        bank_id = int(row.BANKBID)
        cnt = int(row.txn_cnt)

        bank_usage[client_code][bank_id] += cnt
        bank_usage_receiver[client_code][bank_id] += cnt


def make_bank_usage_json(counter: Counter):
    """
    Формує рейтинг банків клієнта у JSON-форматі.
    """
    if not counter:
        return None

    total = sum(counter.values())

    result = []

    for bank_id, cnt in counter.most_common():
        result.append({
            "bank_id": bank_id,
            "txn_cnt": int(cnt),
            "share": round(cnt / total, 4)
        })

    return json.dumps(result, ensure_ascii=False)


def get_top_bank(counter: Counter):
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def get_top_bank_count(counter: Counter):
    if not counter:
        return 0
    return int(counter.most_common(1)[0][1])


def get_top_bank_share(counter: Counter):
    if not counter:
        return 0

    total = sum(counter.values())
    top_cnt = counter.most_common(1)[0][1]

    return round(top_cnt / total, 4)


def get_banks_used_count(counter: Counter):
    return len(counter)


def get_total_txn_count(counter: Counter):
    return int(sum(counter.values()))


sample["BANK_USAGE_JSON"] = sample["IDENTIFYCODE"].map(
    lambda x: make_bank_usage_json(bank_usage.get(x, Counter()))
)

sample["BANK_USAGE_SENDER_JSON"] = sample["IDENTIFYCODE"].map(
    lambda x: make_bank_usage_json(bank_usage_sender.get(x, Counter()))
)

sample["BANK_USAGE_RECEIVER_JSON"] = sample["IDENTIFYCODE"].map(
    lambda x: make_bank_usage_json(bank_usage_receiver.get(x, Counter()))
)

sample["TOP_BANK_ID"] = sample["IDENTIFYCODE"].map(
    lambda x: get_top_bank(bank_usage.get(x, Counter()))
)

sample["TOP_BANK_TXN_CNT"] = sample["IDENTIFYCODE"].map(
    lambda x: get_top_bank_count(bank_usage.get(x, Counter()))
)

sample["TOP_BANK_SHARE"] = sample["IDENTIFYCODE"].map(
    lambda x: get_top_bank_share(bank_usage.get(x, Counter()))
)

sample["BANKS_USED_CNT"] = sample["IDENTIFYCODE"].map(
    lambda x: get_banks_used_count(bank_usage.get(x, Counter()))
)

sample["TOTAL_BANK_TXN_CNT"] = sample["IDENTIFYCODE"].map(
    lambda x: get_total_txn_count(bank_usage.get(x, Counter()))
)


# =========================
# 6. Optional: flag OTP usage
# =========================

sample["OTP_TXN_CNT"] = sample["IDENTIFYCODE"].map(
    lambda x: int(bank_usage.get(x, Counter()).get(MFO_OTP, 0))
)

sample["OTP_TXN_SHARE"] = sample.apply(
    lambda row: round(row["OTP_TXN_CNT"] / row["TOTAL_BANK_TXN_CNT"], 4)
    if row["TOTAL_BANK_TXN_CNT"] > 0 else 0,
    axis=1
)

sample[[
    "IDENTIFYCODE",
    "TOP_BANK_ID",
    "TOP_BANK_TXN_CNT",
    "TOP_BANK_SHARE",
    "BANKS_USED_CNT",
    "TOTAL_BANK_TXN_CNT",
    "OTP_TXN_CNT",
    "OTP_TXN_SHARE",
    "BANK_USAGE_JSON"
]].head(20)