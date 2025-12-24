RE_REFUND = re.compile(
    r"""
    \bвідшк\w*\b |
    \brefund\b |
    \breversal\b
    """,
    re.IGNORECASE | re.VERBOSE
)

# === STRONG INCLUDE: any refund is acquiring ===
if RE_REFUND.search(pp):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "refund",
        "acq_score": 1
    })
