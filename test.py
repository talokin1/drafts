RE_BANK_CP = re.compile(
    r"""
    ощадбанк |
    райффайзен |
    raiffeisen |
    сенс\s*банк |
    sense\s*bank |
    приватбанк |
    otp\s*bank |
    укрсиб |
    пумб |
    monobank
    """,
    re.IGNORECASE | re.VERBOSE
)

# === HARD NEGATIVE: household payments via BANK are NOT acquiring ===
if RE_HOUSEHOLD_NEG.search(pp) and RE_BANK_CP.search(cp):
    # якщо немає явного acquiring-сигналу — стоп
    if not any([
        RE_OPER_ACQ.search(pp),
        RE_REFUND.search(pp),
        RE_COVERAGE.search(pp),
        RE_TYPE_ACQ.search(pp),
        RE_INTERNET_ACQ_CP.search(cp),
        RE_COUNTERPARTY.search(cp),
    ]):
        return pd.Series({
            "is_acquiring": False,
            "acq_reason": "",
            "acq_score": 0
        })
