RE_MERCHANT_CASH = re.compile(
    r"""
    видач\w*\s+
    готівк\w*\s+
    кошт\w* .*?
    держател\w*\s+
    епз
    """,
    re.IGNORECASE | re.VERBOSE
)
