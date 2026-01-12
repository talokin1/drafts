RE_MERCHANT_CASH = re.compile(
    r"""
    видач\w*\s+
    готівк\w*\s+
    кошт\w* .*?
    держател\w*\s+
    епз
    """,
    re.IGNORECASE | re.VERBOSE
)

if RE_MERCHANT_CASH.search(pp_text):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "merchant_cash_withdrawal",
        "acq_score": 3,
    })

