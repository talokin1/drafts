RE_REFUND_CASH = re.compile(
    r"""
    відшк\w* .*?
    видач\w* \s+ готівк\w*
    """,
    re.IGNORECASE | re.VERBOSE
)


if RE_REFUND_CASH.search(pp_text) and (
    RE_CMPS_MERCHANT.search(pp_text)
    or RE_ADDRESS_HINT.search(pp_text)
):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "refund_merchant_cash",
        "acq_score": 3,
    })
