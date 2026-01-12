RE_REFUND_ANY = re.compile(
    r"\bвідшк\w*|\brefund\b",
    re.IGNORECASE
)

RE_CMPS = re.compile(
    r"\bcmps\b",
    re.IGNORECASE
)




RE_REFUND_OPERATIONAL = re.compile(
    r"(відшк\w*).*(опер\.?|операц|торг|видач\w*.*?(готів|гот\.?))",
    re.IGNORECASE
)

RE_CMPS = re.compile(
    r"\bcmps\b",
    re.IGNORECASE
)

