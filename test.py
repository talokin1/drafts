RE_CMPS = re.compile(
    r"\bcmps\b|\bcmp\b|\bcard\s*processing\b",
    re.IGNORECASE
)

RE_MERCHANT_ADDRESS = re.compile(
    r"\b\d{5}\b.*?(вул|просп|пр-т|street|st\.?)",
    re.IGNORECASE
)

RE_SAME_DAY_PERIOD = re.compile(
    r"\b(\d{8})\s*-\s*\1\b"
)


# --- CMPS / card-processing acquiring ---
if (
    RE_CMPS.search(pp_text)
    and RE_COMMISSION.search(pp_text)
    and (
        RE_MERCHANT_ADDRESS.search(pp_text)
        or RE_SAME_DAY_PERIOD.search(pp_text)
    )
):
    cp_reasons.append("cmps_card_processing")
