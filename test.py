RE_REFUND_CASH_STRICT = re.compile(
    r"відшкод\w*.*видач\w*.*гот",
    re.IGNORECASE
)

# 1. HARD OVERRIDE: refund cash = acquiring
if RE_REFUND_CASH_STRICT.search(pp_text):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "refund_cash_acquiring",
        "acq_score": 3,   # навмисно високий
    })
