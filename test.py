import re
import pandas as pd

RE_COMMISSION = re.compile(
    r"\b(cmps|cmp|коміс|ком\.?|komis|commission)\b.*?(\d+[.,]?\d*)",
    re.IGNORECASE
)

RE_CASH = re.compile(r"видач\w*.*?(готів|гот\.?).*?(епз|карт)", re.IGNORECASE)
RE_COVERAGE = re.compile(r"покрит\w*.*?(пк|карт|card)", re.IGNORECASE)
RE_OPER_ACQ = re.compile(r"(опер\.?|операц|торг\.?|торгів).*?екв", re.IGNORECASE)
RE_REFUND = re.compile(r"відшк\w*.*?екв", re.IGNORECASE)
RE_TYPE_ACQ = re.compile(r"type\s*acquir|liqpay|split\s+id", re.IGNORECASE)
RE_COUNTERPARTY = re.compile(r"еквайр|acquir|liqpay", re.IGNORECASE)

RE_INTERNET_ACQ_CP = re.compile(
    r"(інтер[\s\-]*еквайр|internet[\s\-]*acquir|inter[\s\-]*acquir)",
    re.IGNORECASE
)

# ✅ Контекстні маркери еквайрингу (мають бути, щоб "cmps/commission" вважати еквайрингом)
RE_ACQ_CONTEXT = re.compile(
    r"(екв\b|еквайр\w*|acquir\w*|liqpay|split\s+id|type\s*acquir)",
    re.IGNORECASE
)

def detect_acquiring(row):
    pp = row.get("PLATPURPOSE", "") or ""
    cp = row.get("CONTRAGENTSANAME", "") or ""

    text_pp = pp.lower()
    text_cp = cp.lower()
    text = f"{text_pp} {text_cp}"
    reasons = []

    # базові сигнали
    if RE_TYPE_ACQ.search(text):
        reasons.append("type_acquiring")

    if RE_OPER_ACQ.search(text):
        reasons.append("operational_acq")

    if RE_REFUND.search(text):
        reasons.append("acq_refund")

    if RE_COVERAGE.search(text):
        reasons.append("cards_coverage")

    if RE_CASH.search(text):
        reasons.append("cash_epz")

    if RE_INTERNET_ACQ_CP.search(cp):
        reasons.append("internet_acquiring")

    if RE_COUNTERPARTY.search(cp):
        reasons.append("counterparty_name")

    # commission/cmps — додаємо ОКРЕМО, бо це головне джерело FP
    has_commission = bool(RE_COMMISSION.search(text))
    if has_commission:
        reasons.append("commission")

    # ✅ FP-фільтр №1:
    # якщо є commission/cmps у PLATPURPOSE, але в контрагенті НЕМА контексту еквайрингу — це не еквайринг
    # (і загалом commission без контексту еквайрингу не рахуємо)
    has_acq_context_in_cp = bool(RE_ACQ_CONTEXT.search(cp))
    if "commission" in reasons and not has_acq_context_in_cp:
        # прибираємо commission як причину
        reasons = [r for r in reasons if r != "commission"]

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(reasons),
        "acq_score": len(reasons)
    })
