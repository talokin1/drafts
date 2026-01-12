RE_REFUND_ACQ = re.compile(
    r"(відшк\w*|refund).*(екв|acquir|liqpay)",
    re.IGNORECASE
)

RE_REFUND_CASH = re.compile(
    r"(відшк\w*|refund).*(видач\w*.*?(готів|гот\.?)|cash)",
    re.IGNORECASE
)



if RE_INTERNET_ACQ_CP.search(cp) or RE_COUNTERPARTY.search(cp):
        return pd.Series({"is_acquiring": True, "acq_reason": "cp_strong", "acq_score": 1})

# 2) Purpose strong: refund (включає cash refund)
if RE_REFUND_ACQ.search(pp):
    return pd.Series({"is_acquiring": True, "acq_reason": "refund_acq", "acq_score": 1})

if RE_REFUND_CASH.search(pp):
    return pd.Series({"is_acquiring": True, "acq_reason": "refund_cash", "acq_score": 1})