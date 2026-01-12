if RE_REFUND_ACQ.search(pp_text):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "refund_acquiring",
        "acq_score": 1,
    })

if RE_REFUND_CASH.search(pp_text):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "refund_cash",
        "acq_score": 1,
    })