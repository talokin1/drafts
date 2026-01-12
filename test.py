RE_REFUND_STRONG = re.compile(
    r"""
    (?<![a-zа-я0-9])
    (
        відшк(?:од|од\.|одув|одуван\w*)? |
        поверн\w* |
        рефанд\w* |
        refund |
        reversal(?:\s+of)?
    )
    """,
    re.IGNORECASE | re.VERBOSE
)
