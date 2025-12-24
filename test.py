RE_CASH_REFUND = re.compile(
    r"""
    (відшкод|повернен|reversal|refund)
    .*?
    (видач\w*\s*готів|готівк)
    """,
    re.IGNORECASE | re.VERBOSE
)

RE_SETTLEMENT_ACQ = re.compile(
    r"""
    (розрах\w*|розрахунок)
    .*?
    (тт|merchant|tt\s*[a-z0-9_]+)
    """,
    re.IGNORECASE | re.VERBOSE
)

if RE_SETTLEMENT_ACQ.search(pp):
    reasons.append("pp_settlement_acquiring")
    pp_hit = True

    RE_INSTALLMENTS = re.compile(
    r"(оплата\s*частинами|покупк\w*\s*частинами|split\s*pay)",
    re.IGNORECASE
)


# NEGATIVE: cash refund is NOT acquiring
if RE_CASH_REFUND.search(pp):
    return pd.Series({
        "is_acquiring": False,
        "acq_reason": "cash_refund",
        "acq_score": 0
    })

# NEGATIVE: installments are NOT acquiring
if RE_INSTALLMENTS.search(pp):
    return pd.Series({
        "is_acquiring": False,
        "acq_reason": "installments",
        "acq_score": 0
    })

