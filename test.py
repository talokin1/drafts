RE_COUNTERPARTY_IS_BANK = re.compile(
    r"\bбанк\b|bank\b|АТ\s*\".*банк\"|АО\s*\".*банк\"|JSC\s+.*bank",
    re.IGNORECASE
)

# optionally: explicit known bank names (extend as needed)
BANK_NAME_BLACKLIST = {
    "сенс банк", "sense bank", "приватбанк", "ощадбанк", "укрсиббанк", "райффайзен",
    "пумб", "monobank", "укргазбанк", "otp bank"
}


def looks_like_bank_name(s: str) -> bool:
    if not s:
        return False
    t = s.lower()
    if RE_COUNTERPARTY_IS_BANK.search(t):
        return True
    return any(b in t for b in BANK_NAME_BLACKLIST)



is_bank = looks_like_bank_name(cp)

if reasons == ["counterparty_weak"] and is_bank:
    return pd.Series({"is_acquiring": False, "acq_reason": "", "acq_score": 0})
