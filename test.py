import pandas as pd
import re

def norm(x):
    return (x or "").lower()

# ===== VERY BROAD SIGNALS =====

RE_STRONG_TEXT = re.compile(
    r"(екв|еквайр|acquir|acquiring|liqpay|split\s*id|type\s*acquir)",
    re.IGNORECASE
)

RE_CMPS = re.compile(r"\bcmps\b|\bcmp\b", re.IGNORECASE)

RE_CARD_CONTEXT = re.compile(
    r"(пк|карт|card|visa|mastercard)",
    re.IGNORECASE
)

RE_REFUND = re.compile(
    r"(відшк|refund|reimb)",
    re.IGNORECASE
)

RE_COVERAGE = re.compile(
    r"(покрит|settlement)",
    re.IGNORECASE
)

# ===== DETECTOR =====

def detect_acquiring_phase_a(row):
    pp = norm(row.get("PLATPURPOSE"))
    cp = norm(row.get("CONTRAGENTSANAME"))

    text = f"{pp} {cp}"

    reasons = []

    if RE_STRONG_TEXT.search(text):
        reasons.append("explicit_acquiring")

    if RE_CMPS.search(pp):
        reasons.append("cmps_raw")

    if RE_REFUND.search(text):
        reasons.append("refund_like")

    if RE_COVERAGE.search(text):
        reasons.append("coverage_like")

    if RE_CARD_CONTEXT.search(text):
        reasons.append("card_context")

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(reasons),
        "acq_score": len(reasons)
    })

# ===== RUN =====

df = pd.read_excel("transactions.xlsx")

res = df.apply(detect_acquiring_phase_a, axis=1)
df = pd.concat([df, res], axis=1)

print("Detected acquiring (phase A):", df["is_acquiring"].mean().round(3))

df.to_excel("transactions_acquiring_phase_a.xlsx", index=False)
