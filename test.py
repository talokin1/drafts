import re
import pandas as pd

RE_COMMISSION = re.compile(
    r"(cmps|cmp|komis|kom\.?|commission).*(\d+[.,]?\d*)",
    re.IGNORECASE
)

RE_CASH = re.compile(
    r"видач\w*.*?(готів|гот\.?).*?(епз|карт)",
    re.IGNORECASE
)

RE_COVERAGE = re.compile(
    r"покрит\w*.*?(пк|карт|card)",
    re.IGNORECASE
)

RE_OPER_ACQ = re.compile(
    r"(опер\.?|операц|торг\.?|торгів).*?екв",
    re.IGNORECASE
)

RE_REFUND = re.compile(
    r"відшк\w*.*?екв",
    re.IGNORECASE
)

RE_TYPE_ACQ = re.compile(
    r"type\s*acquir|liqpay|split\s*id",
    re.IGNORECASE
)

RE_COUNTERPARTY = re.compile(
    r"еквайр|acquir|liqpay",
    re.IGNORECASE
)

RE_INTERNET_ACQ_CP = re.compile(
    r"(інтер[\s\-]*еквайр|internet[\s\-]*acquir|inter[\s\-]*acquir)",
    re.IGNORECASE
)

RE_PART_PAY = re.compile(
    r"оплата\s*частинами",
    re.IGNORECASE
)

RE_HOUSEHOLD = re.compile(
    r"""
    житлово[\s\-]?комунальн |
    комун\w* |
    ком\.послуг |
    рахунок\s+за\s+сплат |
    оренд\w* |
    паркомісц |
    платник\b |
    \bпн\b
    """,
    re.IGNORECASE | re.VERBOSE
)

RE_BANK_LIKE = re.compile(
    r"""
    \b(at|ao|pat|prat|jsc|pjsc)\b.*?\bбанк\b |
    \bбанк\b.*?\b(at|ao|pat|prat|jsc|pjsc)\b |
    ощад |
    райф
    """,
    re.IGNORECASE | re.VERBOSE
)



def detect_acquiring(row: pd.Series) -> pd.Series:
    pp_text = (row.get("PLATPURPOSE") or "").lower()
    cp_text = (row.get("CONTRAGENTASNAME") or "").lower()

    if RE_PART_PAY.search(pp_text):
        return pd.Series({
            "is_acquiring": False,
            "acq_reason": "",
            "acq_score": 0,
        })

    if (
        RE_HOUSEHOLD.search(pp_text)
        and RE_BANK_LIKE.search(pp_text)
        and not RE_OPER_ACQ.search(pp_text)
    ):
        return pd.Series({
            "is_acquiring": False,
            "acq_reason": "",
            "acq_score": 0,
        })

 
    cp_reasons = []

    if RE_INTERNET_ACQ_CP.search(cp_text):
        cp_reasons.append("cp_internet_acquiring")

    if RE_COUNTERPARTY.search(cp_text):
        cp_reasons.append("cp_acquiring_keyword")

    if cp_reasons:
        return pd.Series({
            "is_acquiring": True,
            "acq_reason": "|".join(cp_reasons),
            "acq_score": len(cp_reasons),
        })


    pp_reasons = []

    if RE_TYPE_ACQ.search(pp_text):
        pp_reasons.append("type_acquiring")

    if RE_OPER_ACQ.search(pp_text):
        pp_reasons.append("operational_acq")

    if RE_COMMISSION.search(pp_text):
        if RE_OPER_ACQ.search(pp_text) or RE_TYPE_ACQ.search(pp_text):
            pp_reasons.append("commission_acq")

    if RE_COVERAGE.search(pp_text):
        pp_reasons.append("cards_coverage")

    if RE_REFUND.search(pp_text):
        pp_reasons.append("acq_refund")

    if RE_CASH.search(pp_text):
        pp_reasons.append("cash_epz")

    return pd.Series({
        "is_acquiring": len(pp_reasons) > 0,
        "acq_reason": "|".join(pp_reasons),
        "acq_score": len(pp_reasons),
    })
