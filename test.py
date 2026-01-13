RE_REFUND_NON_ACQ = re.compile(
    r"\b("
    r"подат(ок|ку|ки)|"
    r"земел\w*|"
    r"жкг|"
    r"водопостачан\w*|"
    r"електроенерг\w*|"
    r"газопостачан\w*|"
    r"утриманн\w*|"
    r"комунальн\w*|"
    r"оренд\w*"
    r")\b",
    re.IGNORECASE
)

RE_REFUND_ACQ = re.compile(
    r"\b("
    r"еквайр\w*|"
    r"торговельн\w*|"
    r"merchant|"
    r"cmps|"
    r"kasa|"
    r"pos|"
    r"термінал\w*|"
    r"ком\s*бан|"
    r"коміс\w*"
    r")\b",
    re.IGNORECASE
)
