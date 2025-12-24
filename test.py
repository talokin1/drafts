RE_CMPS_COMMISSION_STRONG = re.compile(
    r"\bкоміс(ія|iя|ія)\b|\bкоміс\b",
    re.IGNORECASE
)
pp_has_strong_acq = any([
    RE_OPER_ACQ.search(pp),
    RE_REFUND.search(pp),
    RE_COVERAGE.search(pp),
    RE_TYPE_ACQ.search(pp),
    RE_CASH.search(pp),
    (RE_CMPS.search(pp) and RE_CMPS_COMMISSION_STRONG.search(pp)),
])
