RE_BANK_FEE = re.compile(
    r"\bkom\.?\s*banka\b|\bком\.?\s*банка\b",
    re.IGNORECASE
)


if RE_COMMISSION.search(text) and not RE_OPER_ACQ.search(text):
    # це просто комісія банку

if RE_COMMISSION.search(text) and not (
    RE_OPER_ACQ.search(text) or
    RE_TYPE_ACQ.search(text)
):
    return False
