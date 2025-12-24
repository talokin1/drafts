RE_HOUSEHOLD_NEG = re.compile(
    r"""
    житлово[\-\s]?комунальн |
    комунальн\w* |
    рахунок\s+за\s+сплат |
    оренда |
    паркомісц |
    платник\s+[А-ЯІЇЄA-Z][а-яіїєa-z]+ |
    іпн\s*\d{8,10}
    """,
    re.IGNORECASE | re.VERBOSE
)

# === NEGATIVE: household payments via bank (NOT acquiring) ===
if RE_HOUSEHOLD_NEG.search(pp):
    # якщо немає явного еквайрингу — відсікаємо
    if not any([
        RE_OPER_ACQ.search(pp),
        RE_REFUND.search(pp),
        RE_COVERAGE.search(pp),
        RE_TYPE_ACQ.search(pp),
        RE_INTERNET_ACQ_CP.search(cp),
        RE_COUNTERPARTY.search(cp),
    ]):
        return pd.Series({
            "is_acquiring": False,
            "acq_reason": "",
            "acq_score": 0
        })
