RE_CMPS = re.compile(
    r"\b(cmps|cmp)\b",
    re.IGNORECASE
)

RE_CMPS_CONTEXT = re.compile(
    r"""
    (відшк\w*|екв\b|еквайр\w*|
     покрит\w*|коміс\w*|ком\.?\s*бан|
     к-?ть\s*тр|кільк\w*\s*тр|
     acquir\w*)
    """,
    re.IGNORECASE | re.VERBOSE
)
def detect_acquiring(row):
    pp = row.get("PLATPURPOSE", "") or ""
    cp = row.get("CONTRAGENTSANAME", "") or ""

    text = f"{pp} {cp}".lower()
    reasons = []

    if RE_TYPE_ACQ.search(text):
        reasons.append("type_acquiring")

    if RE_OPER_ACQ.search(text):
        reasons.append("operational_acq")

    if RE_REFUND.search(text):
        reasons.append("acq_refund")

    if RE_COVERAGE.search(text):
        reasons.append("cards_coverage")

    # ✅ cmps only with context
    if RE_CMPS.search(pp) and RE_CMPS_CONTEXT.search(text):
        reasons.append("commission")

    if RE_CASH.search(text):
        reasons.append("cash_epz")

    if RE_INTERNET_ACQ_CP.search(cp):
        reasons.append("internet_acquiring")

    if RE_COUNTERPARTY.search(cp):
        reasons.append("counterparty_name")

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(reasons),
        "acq_score": len(reasons)
    })
