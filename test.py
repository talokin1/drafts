import re

RE_SPACES = re.compile(r"\s+")
RE_QUOTES = re.compile(r"[\"'“”«»„”`´]")  # різні типи лапок

def normalize(s: str) -> str:
    s = (s or "")
    s = s.lower()
    s = RE_QUOTES.sub("", s)          # прибираємо лапки
    s = s.replace("’", "").replace("`", "")
    s = RE_SPACES.sub(" ", s).strip() # нормалізуємо пробіли
    return s

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

def detect_acquiring(row):
    pp_raw = row.get("PLATPURPOSE", "") or ""
    cp_raw = row.get("CONTRAGENTSANAME", "") or ""

    pp = normalize(pp_raw)
    cp = normalize(cp_raw)
    text = f"{pp} {cp}"

    # 0) Негатив: household + bank-like + немає acquiring-маркерів -> не еквайринг
    if RE_HOUSEHOLD.search(pp) and RE_BANK_LIKE.search(cp) and not RE_ACQ_MARKERS.search(text):
        return pd.Series({"is_acquiring": False, "acq_reason": "", "acq_score": 0})

    reasons = []

    # 1) PLATPURPOSE first (як ти хочеш)
    pp_hit = False

    if RE_TYPE_ACQ.search(pp):
        reasons.append("pp_type_acquiring"); pp_hit = True
    if RE_OPER_ACQ.search(pp):
        reasons.append("pp_operational_acq"); pp_hit = True
    if RE_REFUND.search(pp):
        reasons.append("pp_acq_refund"); pp_hit = True
    if RE_COVERAGE.search(pp):
        reasons.append("pp_cards_coverage"); pp_hit = True
    if RE_CASH.search(pp):
        reasons.append("pp_cash_epz"); pp_hit = True

    # cmps guard: cmps тільки з контекстом (у pp або в text)
    if RE_CMPS.search(pp) and RE_CMPS_CONTEXT.search(text):
        reasons.append("pp_cmps_confirmed"); pp_hit = True

    # якщо pp щось дав — повертаємо
    if pp_hit:
        return pd.Series({"is_acquiring": True, "acq_reason": "|".join(reasons), "acq_score": len(reasons)})

    # 2) fallback на CONTRAGENTSANAME
    if RE_INTERNET_ACQ_CP.search(cp):
        reasons.append("cp_internet_acquiring")
    if RE_COUNTERPARTY.search(cp):
        reasons.append("cp_counterparty_name")

    return pd.Series({"is_acquiring": len(reasons) > 0, "acq_reason": "|".join(reasons), "acq_score": len(reasons)})
