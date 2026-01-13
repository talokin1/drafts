RE_REFUND_UTILITIES = re.compile(
    r"\b("
    r"електроенерг\w*|"
    r"ел\.?енерг\w*|"
    r"е/енерг\w*|"
    r"теплопостачан\w*|"
    r"водопостачан\w*|"
    r"реактивн\w*|"
    r"газопостачан\w*|"
    r"вивезенн\w*|"
    r"тпв|"
    r"жкг"
    r")\b",
    re.IGNORECASE
)

text = normalize_ua(pp_text)

if RE_REFUND_BASE.search(text) and (
    RE_REFUND_NON_ACQ_STRONG.search(text)
    or RE_REFUND_UTILITIES.search(text)
    or RE_REFUND_TAX.search(text)
):
    is_acquiring = False
    reason = "refund_non_acquiring"
