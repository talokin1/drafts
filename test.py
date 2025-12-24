RE_INSTALLMENTS = re.compile(
    r"оплата\s*частинами|частинами",
    re.IGNORECASE
)

if RE_INSTALLMENTS.search(pp):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "installments_payment",
        "acq_score": 1
    })



RE_REFUND = re.compile(
    r"""
    відшк\w*
    .*?
    (екв|видач\w*\s*готів)
    """,
    re.IGNORECASE | re.VERBOSE
)
