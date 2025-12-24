import pandas as pd
import re

# =========================
# Regex patterns (generalized)
# =========================

RE_COMMISSION = re.compile(
    r"(cmps|cmp|коміс|ком\.?|komis|commission|ком\s*бан).*?(\d+[.,]?\d*)",
    re.IGNORECASE
)

RE_CASH = re.compile(
    r"видач\w*.*?(готів|гот\.?).*?(епз|карт)",
    re.IGNORECASE
)

RE_COVERAGE = re.compile(
    r"покрит\w*.*?(пк|карт|card)",
    re.IGNORECASE
)

RE_OPER_ACQ = re.compile(
    r"(опер\.?|операц|торг\.?|торгів).*?екв",
    re.IGNORECASE
)

RE_REFUND = re.compile(
    r"(відшк|відшкод)\w*.*?(екв|еквайр)",
    re.IGNORECASE
)

# explicit acquiring markers (incl. LIQPAY SPLIT ID / TYPE acquiring)
RE_TYPE_ACQ = re.compile(
    r"(type\s*acquir|split\s*id|liqpay\s*split|pbk\b.*type\s*acquir)",
    re.IGNORECASE
)

# -------------------------
# Counterparty-name patterns
# -------------------------

# your missing case: "Плат.інтер-еквайринг через LiqPay"
RE_COUNTERPARTY_INTER_ACQ = re.compile(
    r"(інтер[\s\-]*еквайр|inter[\s\-]*acquir|еквайр).*?(liqpay|liq\s*pay)",
    re.IGNORECASE
)

# generic acquiring words in counterparty (weak signal)
RE_COUNTERPARTY_ACQ_WEAK = re.compile(
    r"(еквайр|acquir|liqpay)",
    re.IGNORECASE
)

# bank-like counterparty names (to avoid false positives)
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

# =========================
# Detection
# =========================

def detect_acquiring(row):
    pp = (row.get("PLATPURPOSE", "") or "")
    cp = (row.get("CONTRAGENTSANAME", "") or "")

    text = f"{pp} {cp}".lower()

    reasons = []

    # Strong rules first
    if RE_TYPE_ACQ.search(text):
        reasons.append("type_acquiring")

    if RE_COUNTERPARTY_INTER_ACQ.search(cp):
        reasons.append("counterparty_inter_acquiring")

    # Medium rules
    if RE_OPER_ACQ.search(text):
        reasons.append("operational_acq")

    if RE_REFUND.search(text):
        reasons.append("acq_refund")

    if RE_COVERAGE.search(text):
        reasons.append("cards_coverage")

    if RE_COMMISSION.search(text):
        reasons.append("commission")

    if RE_CASH.search(text):
        reasons.append("cash_epz")

    # Weak fallback (counterparty mentions acquiring)
    if RE_COUNTERPARTY_ACQ_WEAK.search(cp):
        reasons.append("counterparty_weak")

    # -------------------------
    # Anti-false-positive guard:
    # if ONLY weak counterparty rule triggered and counterparty is a bank -> ignore
    # -------------------------
    is_bank = looks_like_bank_name(cp)

    if reasons == ["counterparty_weak"] and is_bank:
        return pd.Series({"is_acquiring": False, "acq_reason": "", "acq_score": 0})

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(reasons),
        "acq_score": len(reasons)
    })


# =========================
# Run pipeline
# =========================

df = pd.read_excel("transactions.xlsx")  # columns: PLATPURPOSE, CONTRAGENTSANAME

detected = df.apply(detect_acquiring, axis=1)
df = pd.concat([df, detected], axis=1)

print("Detected acquiring:", int(df["is_acquiring"].sum()))
print("Detected %:", round(df["is_acquiring"].mean() * 100, 2), "%")

print("\nTop reasons:")
print(
    df[df["is_acquiring"]]
    .groupby("acq_reason")
    .size()
    .sort_values(ascending=False)
    .head(30)
)

df.to_excel("transactions_with_acquiring_flag.xlsx", index=False)
