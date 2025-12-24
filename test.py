RE_BANK_CP = re.compile(
    r"""
    \bбанк\b |                      # будь-який "банк"
    raiffeisen | райффайзен | райф | # Райффайзен
    ощад | oschad |                 # Ощад
    сенс | sense |                  # Sense
    приват | privat |               # Приват
    укрсиб | ukrsib |               # Укрсиб
    укргаз | ukrgas |               # Укргаз
    пумб | pumb |                   # PUMB
    mono | monobank |               # mono
    otp\b | otp\s*bank              # OTP
    """,
    re.IGNORECASE | re.VERBOSE
)


# HARD STOP: household payments to BANK are NOT acquiring
if RE_HOUSEHOLD_NEG.search(pp) and RE_BANK_CP.search(cp):
    # виняток: якщо є явний acquiring-сигнал у cp/pp (liqpay/type acquiring/екв)
    if not any([
        RE_TYPE_ACQ.search(pp) or RE_TYPE_ACQ.search(cp),
        RE_INTERNET_ACQ_CP.search(cp),
        re.search(r"\bекв\b|еквайр|acquir|liqpay|split\s*id|type\s*acquir", cp, re.IGNORECASE),
        re.search(r"\bекв\b|еквайр|acquir|liqpay|split\s*id|type\s*acquir", pp, re.IGNORECASE),
    ]):
        return pd.Series({"is_acquiring": False, "acq_reason": "", "acq_score": 0})
