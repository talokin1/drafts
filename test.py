RE_REFUND_STRONG = re.compile(
    r"""
    \b(
        відшк\w*        |  # відшкод, відшкодування
        поверн\w*       |  # повернення
        рефанд\w*       |  # рефанд
        refund          |
        reversal        |
        reversal\s+of
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)


if RE_REFUND_STRONG.search(pp_text):
    # acquiring-refund тільки якщо є хоча б один acquiring-якір
    anchors = 0

    if RE_COMMISSION.search(pp_text):
        anchors += 1
    if RE_COUNTERPARTY.search(cp_text):
        anchors += 1
    if RE_INTERNET_ACQ_CP.search(cp_text):
        anchors += 1
    if RE_CMPS_MERCHANT.search(pp_text):
        anchors += 1
    if RE_ADDRESS_HINT.search(pp_text):
        anchors += 1

    if anchors > 0:
        return pd.Series({
            "is_acquiring": True,
            "acq_reason": "acq_refund",
            "acq_score": 2 + anchors,  # підсилюємо, але не максимум
        })
