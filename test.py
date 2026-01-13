RE_INSTALLMENTS = re.compile(
    r"""
    оплата\s*(частин(ами|ах)?|по\s*частинам) |
    розрахунк(и|і)\s*["«]?\s*оплата\s*частин(ами|ах)?\s*["»]? |
    installment(s)? |
    payment\s*in\s*installments |
    oplata\s*chastyn(am|amy|akh)?
    """,
    re.IGNORECASE | re.VERBOSE
)
