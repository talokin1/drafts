if re.search(r"розр\w*", pp_text):
    anchors = 0

    if RE_COMMISSION.search(pp_text):
        anchors += 1
    if re.search(r"\bтт\b", pp_text):
        anchors += 1
    if re.search(r"реєстр", pp_text):
        anchors += 1
    if RE_CMPS_MERCHANT.search(pp_text):
        anchors += 1

    if anchors >= 2:
        return pd.Series({
            "is_acquiring": True,
            "acq_reason": "acq_settlement",
            "acq_score": 2 + anchors,
        })
