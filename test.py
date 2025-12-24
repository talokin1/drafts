import pandas as pd
import re

RE_CMPS = re.compile(r"\b(cmps|cmp)\b", re.IGNORECASE)

RE_CMPS_CONTEXT = re.compile(
    r"""
    (відшк\w*|екв\b|еквайр\w*|
     покрит\w*|коміс\w*|ком\.?\s*бан|
     к-?ть\s*тр|кільк\w*\s*тр|
     acquir\w*)
    """,
    re.IGNORECASE | re.VERBOSE
)

RE_CASH = re.compile(
    r"видач\w*.*?(готів|гот\.?).*?(епз|карт)",
    re.IGNORECASE
)

RE_COVERAGE = re.compile(
    r"покрит\w*.*?(пк|карт|card)",
    re.IGNORECASE
)

RE_OPER_ACQ = re.compile(
    r"(опер\.?|операц|торг\.?|торгів).*?екв",
    re.IGNORECASE
)

RE_REFUND = re.compile(
    r"відшк\w*.*?екв",
    re.IGNORECASE
)

RE_TYPE_ACQ = re.compile(
    r"type\s*acquir|liqpay|split\s+id",
    re.IGNORECASE
)

RE_COUNTERPARTY = re.compile(
    r"еквайр|acquir|liqpay",
    re.IGNORECASE
)

RE_INTERNET_ACQ_CP = re.compile(
    r"(інтер[\s\-]*еквайр|internet[\s\-]*acquir|inter[\s\-]*acquir)",
    re.IGNORECASE
)

def detect_acquiring(row):
    pp = (row.get("PLATPURPOSE", "") or "").lower()
    cp = (row.get("CONTRAGENTSANAME", "") or "").lower()

    reasons = []

    # =========================
    # 1️⃣ ПРІОРИТЕТ: PLATPURPOSE
    # =========================

    pp_hit = False

    if RE_TYPE_ACQ.search(pp):
        reasons.append("pp_type_acquiring")
        pp_hit = True

    if RE_OPER_ACQ.search(pp):
        reasons.append("pp_operational_acq")
        pp_hit = True

    if RE_REFUND.search(pp):
        reasons.append("pp_acq_refund")
        pp_hit = True

    if RE_COVERAGE.search(pp):
        reasons.append("pp_cards_coverage")
        pp_hit = True

    if RE_CASH.search(pp):
        reasons.append("pp_cash_epz")
        pp_hit = True

    # cmps ТІЛЬКИ З КОНТЕКСТОМ і ТІЛЬКИ В PLATPURPOSE
    if RE_CMPS.search(pp) and RE_CMPS_CONTEXT.search(pp):
        reasons.append("pp_cmps_confirmed")
        pp_hit = True

    # ❗ Якщо щось знайшли в PLATPURPOSE — ПОВЕРТАЄМОСЬ
    if pp_hit:
        return pd.Series({
            "is_acquiring": True,
            "acq_reason": "|".join(reasons),
            "acq_score": len(reasons)
        })

    # ==================================
    # 2️⃣ FALLBACK: CONTRAGENTSANAME
    # ==================================

    if RE_INTERNET_ACQ_CP.search(cp):
        reasons.append("cp_internet_acquiring")

    if RE_COUNTERPARTY.search(cp):
        reasons.append("cp_counterparty_acq")

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(reasons),
        "acq_score": len(reasons)
    })

df = pd.read_excel("transactions.xlsx")

detected = df.apply(detect_acquiring, axis=1)
df = pd.concat([df, detected], axis=1)

print("Знайдено еквайрингу:", df["is_acquiring"].sum())
print("Відсоток:", round(df["is_acquiring"].mean() * 100, 2), "%")

print(
    df[df["is_acquiring"]]
    .groupby("acq_reason")
    .size()
    .sort_values(ascending=False)
)

df.to_excel("transactions_with_acquiring_flag.xlsx", index=False)
