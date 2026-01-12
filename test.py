full_text = f"{pp_text} {cp_text}"

# 🚨 HARD RULE: CMPS = acquiring
if RE_CMPS_STRICT.search(full_text):
    return pd.Series({
        "is_acquiring": True,
        "acq_reason": "cmps_commission_acquiring",
        "acq_score": 3,
    })



RE_CMPS_STRICT = re.compile(
    r"\bcmps\b",
    re.IGNORECASE
)
