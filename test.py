def detect_acquiring(row):
    pp = row.get("PLATPURPOSE", "") or ""
    cp = row.get("CONTRAGENTSANAME", "") or ""

    text = f"{pp} {cp}".lower()
    reasons = []

    # --- flags ---
    pp_has_cmps = RE_CMPS.search(pp) is not None

    pp_has_strong_acq = any([
        RE_OPER_ACQ.search(pp),
        RE_REFUND.search(pp),
        RE_COVERAGE.search(pp),
        RE_TYPE_ACQ.search(pp),
        RE_CASH.search(pp),
    ])

    # --- main rules ---
    if RE_TYPE_ACQ.search(text):
        reasons.append("type_acquiring")

    if RE_OPER_ACQ.search(text):
        reasons.append("operational_acq")

    if RE_REFUND.search(text):
        reasons.append("acq_refund")

    if RE_COVERAGE.search(text):
        reasons.append("cards_coverage")

    if RE_COMMISSION.search(text):
        reasons.append("commission")

    if RE_CASH.search(text):
        reasons.append("cash_epz")

    if RE_INTERNET_ACQ_CP.search(cp):
        reasons.append("internet_acquiring")

    if RE_COUNTERPARTY.search(cp):
        reasons.append("counterparty_name")

    # === 🔴 CMPS GUARD (єдина нова логіка) ===
    if pp_has_cmps and not pp_has_strong_acq:
        cp_has_acq = any([
            RE_INTERNET_ACQ_CP.search(cp),
            RE_COUNTERPARTY.search(cp),
            RE_TYPE_ACQ.search(cp),
        ])
        if not cp_has_acq:
            return pd.Series({
                "is_acquiring": False,
                "acq_reason": "",
                "acq_score": 0
            })

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(reasons),
        "acq_score": len(reasons)
    })
