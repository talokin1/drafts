RE_OPERATIONAL_REFUND = re.compile(
    r"(відшкод\w*|refund).*(коміс|fee|corr|adjust)",
    re.IGNORECASE
)

if RE_OPERATIONAL_REFUND.search(full_text):
    return {
        "is_acquiring": False,
        "reason": "operational_refund",
        "score": 0
    }
