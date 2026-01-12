import re
import pandas as pd

# Збережені оригінальні патерни з вашими змінами для гнучкості
RE_COMMISSION = re.compile(
    r"(cmps|cmp|komis |kom\. ?| commission)",  # Видалив обов'язкове число в кінці, щоб ловити більше
    re.IGNORECASE
)

RE_CASH = re.compile(
    r"відшкод\w* ?(ротіб|rot\. ?.)* ?(ен|карт)",
    re.IGNORECASE
)

RE_COVERAGE = re.compile(
    r"покрит\w* ?(nk| карт |card)",
    re.IGNORECASE
)

RE_OPER_ACQ = re.compile(
    r"(onep\. ?|onepay| topр \.?|toprib).*?eka",
    re.IGNORECASE
)

RE_REFUND = re.compile(
    r"відшкод\w* ?екв",
    re.IGNORECASE
)

RE_TYPE_ACQ = re.compile(
    r"type\s*acquir|liqpay|split\s*id",
    re.IGNORECASE
)

RE_COUNTERPARTY = re.compile(
    r"еквайр|acquir|liqpay|fondy|wayforpay|portmone|interkas|concord|ipay|paypong",  # Додав популярні провайдери для більше покриття
    re.IGNORECASE
)

RE_INTERNET_ACQ_CP = re.compile(
    r"(інтерп|internet)\s*.*?еквайр|internet\s*-*acquir|inter\s*.*?acquir",  # Трохи розширив для варіацій
    re.IGNORECASE
)

RE_PART_PAY = re.compile(
    r"оплата\s*частин",
    re.IGNORECASE
)

RE_HOUSEHOLD = re.compile(
    r"хтлово| \s- ?|комунальн|комун| послу|рахунок\s+за\s+сп|операц|паркоміс|платник в|внш",
    re.IGNORECASE | re.VERBOSE
)

RE_BANK_LIKE = re.compile(
    r"b(at |ao| pat| prat |jsc |pjs c)\b ?b|вбанк\b ?b| b(at|ao| pat| prat |jsc |pjsc)\b",
    re.IGNORECASE | re.VERBOSE
)

RE_REFUND_ACQ = re.compile(
    r"(відшкод\w*| refund ). (екв|acquir|liqpay)",
    re.IGNORECASE
)

RE_REFUND_CASH_STRICT = re.compile(
    r"відшкод\w*\s* ? \s* ? no\s*відшкод\w* \s* ?rotib",
    re.IGNORECASE
)

# Нові патерни для зменшення FN
RE_GENERAL_ACQ = re.compile(
    r"(еквайр|acquir|pos|термінал|картков|платіжн карт|visa|mastercard|prostir|розрахунок за карт|надходж за екв)",  # Загальні фрази з виписок
    re.IGNORECASE
)

RE_SETTLEMENT = re.compile(
    r"(розрахунок|settlement|надходж|зарахув).*(еквайр|acquir|карт|card|платіж|pos)",  # Для розрахунків/надходжень
    re.IGNORECASE
)

def detect_acquiring(row: pd.Series) -> pd.Series:
    pp_text = (row.get("PLATFORMPURPOSE") or "").lower()
    cp_text = (row.get("CONTRAGENTASNAME") or "").lower()

    # Ранні виключення (зберіг оригінал)
    if RE_REFUND_ACQ.search(pp_text):
        return pd.Series({
            "is_acquiring": True,
            "acq_reason": "refund acquiring",
            "acq_score": 1,
        })

    if RE_REFUND_CASH_STRICT.search(pp_text):
        return pd.Series({
            "is_acquiring": True,
            "acq_reason": "refund cash acquiring",
            "acq_score": 3,
        })

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

    # CP перевірки (розширені)
    cp_reasons = []
    if RE_INTERNET_ACQ_CP.search(cp_text):
        cp_reasons.append("cp_internet_acquiring")
    if RE_COUNTERPARTY.search(cp_text):
        cp_reasons.append("cp_acquiring_keyword")
    if RE_GENERAL_ACQ.search(cp_text):  # Додав нову перевірку для CP
        cp_reasons.append("cp_general_acq")

    if cp_reasons:
        return pd.Series({
            "is_acquiring": True,
            "acq_reason": " | ".join(cp_reasons),
            "acq_score": len(cp_reasons),
        })

    # PP перевірки (розширені)
    pp_reasons = []
    if RE_TYPE_ACQ.search(pp_text):
        pp_reasons.append("type_acquiring")
    if RE_OPER_ACQ.search(pp_text):
        pp_reasons.append("operational_acq")
    if RE_COMMISSION.search(pp_text):
        # Зробив менш жорстким: append навіть без OPER/TYPE, якщо є general acq
        if RE_OPER_ACQ.search(pp_text) or RE_TYPE_ACQ.search(pp_text) or RE_GENERAL_ACQ.search(pp_text):
            pp_reasons.append("commission_acq")
    if RE_COVERAGE.search(pp_text):
        pp_reasons.append("cards_coverage")
    if RE_REFUND.search(pp_text):
        pp_reasons.append("acq_refund")
    if RE_CASH.search(pp_text):
        pp_reasons.append("cash epz")
    if RE_GENERAL_ACQ.search(pp_text):  # Нова
        pp_reasons.append("general_acq")
    if RE_SETTLEMENT.search(pp_text):  # Нова
        pp_reasons.append("settlement_acq")

    if pp_reasons:
        return pd.Series({
            "is_acquiring": True,
            "acq_reason": " | ".join(pp_reasons),
            "acq_score": len(pp_reasons),
        })

    # Якщо нічого не зматчило
    return pd.Series({
        "is_acquiring": False,
        "acq_reason": "",
        "acq_score": 0,
    })