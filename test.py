RE_REFUND_OPERATIONAL = re.compile(
    r"(відшк\w*).*(опер\.?|операц|торг|видач\w*.*?(готів|гот\.?))",
    re.IGNORECASE
)

RE_CMPS = re.compile(
    r"\bcmps\b",
    re.IGNORECASE
)


# =========================
# STRONG refund (operational)
# =========================
if RE_REFUND_OPERATIONAL.search(pp_text):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "refund_operational",
        "acq_score": 1,
    })

# =========================
# STRONG cmps (merchant acquiring)
# =========================
if RE_CMPS.search(pp_text):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "cmps_merchant",
        "acq_score": 1,
    })
