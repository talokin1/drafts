RE_HOUSEHOLD = re.compile(
    r"""
    житлово[\-\s]?комунальн |
    комун\w* |
    ком\.послуг |
    рахунок\s+за\s+сплат |
    оренд\w* |
    паркомісц |
    платник\b |
    \bіпн\b
    """,
    re.IGNORECASE | re.VERBOSE
)

RE_ACQ_MARKERS = re.compile(
    r"""
    \bекв\b|еквайр\w*|acquir\w*|
    liqpay|split\s*id|type\s*acquir|
    відшк\w*|покрит\w*
    """,
    re.IGNORECASE | re.VERBOSE
)

RE_BANK_LIKE = re.compile(
    r"""
    \b(ат|ао|пат|прат|jsc|pjsc)\b      # організаційна форма (може бути)
    .*?                               # будь-що між
    \bбанк\b                          # слово банк
    |
    \bбанк\b.*?\b(ат|ао|пат|прат|jsc|pjsc)\b
    """,
    re.IGNORECASE | re.VERBOSE
)



if RE_HOUSEHOLD.search(pp) and RE_BANK_LIKE.search(cp) and not RE_ACQ_MARKERS.search(text):
        return pd.Series({"is_acquiring": False, "acq_reason": "", "acq_score": 0})