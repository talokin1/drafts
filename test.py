RE_CP_ACQ_STRONG = re.compile(
    r"""
    розрахунк\w*\s+з\s+еквайр\w* |
    еквайринг |
    acquiring
    """,
    re.IGNORECASE | re.VERBOSE
)

def detect_acquiring(row):
    pp = row.get("PLATPURPOSE", "") or ""
    cp = row.get("CONTRAGENTSANAME", "") or ""

    text = f"{pp} {cp}".lower()
    reasons = []

    # === STRONG: counterparty name ===
    if RE_CP_ACQ_STRONG.search(cp):
        reasons.append("counterparty_acquiring")

    # === OTHER STRONG ===
    if RE_TYPE_ACQ.search(text):
        reasons.append("type_acquiring")

    if RE_OPER_ACQ.search(text):
        reasons.append("operational_acq")

    if RE_REFUND.search(text):
        reasons.append("acq_refund")

    if RE_COVERAGE.search(text):
        reasons.append("cards_coverage")

    # === CMPS: only if CP confirms acquiring ===
    if RE_CMPS.search(pp):
        if RE_CP_ACQ_STRONG.search(cp) or RE_CMPS_CONTEXT.search(text):
            reasons.append("cmps_confirmed")

    if RE_CASH.search(text):
        reasons.append("cash_epz")

    if RE_INTERNET_ACQ_CP.search(cp):
        reasons.append("internet_acquiring")

    if RE_COUNTERPARTY.search(cp):
        reasons.append("counterparty_name")

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(dict.fromkeys(reasons)),  # прибираємо дублікати
        "acq_score": len(set(reasons))
    })
