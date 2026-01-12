RE_CMPS_MERCHANT = re.compile(
    r"\bcmps\b.*коміс",
    re.IGNORECASE
)

RE_ADDRESS_HINT = re.compile(
    r"\b(київ|харків|львів|одеса|дніпро|вул\.?|пр\.?|просп|шосе)\b",
    re.IGNORECASE
)

if (
    RE_CMPS_MERCHANT.search(pp_text)
    and RE_ADDRESS_HINT.search(pp_text)
):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "cmps_merchant_commission",
        "acq_score": 3,
    })
