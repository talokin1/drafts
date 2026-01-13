RE_TRANSPORT_REIMBURSEMENT = re.compile(
    r"""
    \b
    відшк\w*            # відшкодування / відшкод.
    \s+
    транспортн\w*       # транспортних
    \s+
    перевез\w*          # перевезень
    \b
    """,
    re.IGNORECASE | re.VERBOSE
)
