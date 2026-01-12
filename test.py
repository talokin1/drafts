is_refund_cash = RE_REFUND_CASH_STRICT.search(full_text)
is_refund_acq  = RE_REFUND_ACQ.search(full_text)


if is_refund_cash:
    return {
        "is_acquiring": True,
        "reason": "refund_cash_acquiring",
        "score": 3
    }

if is_refund_acq:
    return {
        "is_acquiring": True,
        "reason": "refund_acquiring",
        "score": 2
    }



if (
    RE_OPERATIONAL_REFUND.search(full_text)
    and not (is_refund_cash or is_refund_acq)
):
    return {
        "is_acquiring": False,
        "reason": "operational_refund",
        "score": 0
    }
