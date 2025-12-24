import pandas as pd
import re

# =========================
# Helpers
# =========================

def _norm(x: str) -> str:
    return (x or "").lower().strip()

def _join_text(pp: str, cp: str) -> str:
    return f"{_norm(pp)} {_norm(cp)}"

# =========================
# POSITIVE patterns (strong / medium)
# =========================

# 6) TYPE acquiring / LIQPAY SPLIT / explicit acquiring markers
RE_TYPE_ACQ = re.compile(r"\btype\s*acquir\w*\b|\bliqpay\b.*\bsplit\b|\bsplit\s*id\b", re.IGNORECASE)

# 4) operational acquiring (опер/торг + екв)
RE_OPER_ACQ = re.compile(r"(опер\.?|оператив\w*|операц\w*|торг\.?|торгів\w*).{0,40}\bекв\b", re.IGNORECASE)

# 5) refund acquiring (відшк + екв/еквайр)
RE_REFUND = re.compile(r"(відшк\w*|відшкод\w*).{0,60}(екв\b|еквайр\w*)", re.IGNORECASE)

# 3) coverage cards (покрит + пк/карт)
RE_COVERAGE = re.compile(r"покрит\w*.{0,60}(пк\b|карт\w*|card\w*)", re.IGNORECASE)

# 2) cash withdrawal epz
RE_CASH = re.compile(r"видач\w*.{0,60}(готів|гот\.?).{0,60}(епз\b|карт\w*)", re.IGNORECASE)

# 7) counterparty name contains acquiring (STRONG phrases only)
# - includes your missing case: "Розрахунки з еквайрингу"
RE_CP_STRONG = re.compile(
    r"(розрахунк\w*\s+з\s+еквайр\w*|інтер[\s\-]*еквайр\w*|inter[\s\-]*acquir\w*|internet[\s\-]*acquir\w*)",
    re.IGNORECASE
)

# =========================
# CMPS handling (weak signal, needs confirmation)
# =========================

RE_CMPS = re.compile(r"\bcmps\b|\bcmp\b", re.IGNORECASE)

# CMPS-confirmation keywords: if CMPS exists, we require at least one of these nearby/anywhere
RE_CMPS_CONFIRM = re.compile(
    r"\bекв\b|еквайр\w*|відшк\w*|покрит\w*|коміс\w*|ком\s*бан|к-?ть\s*тр|кільк\w*\s*тр|acquir\w*",
    re.IGNORECASE
)

# Explicit CMPS negative contexts (your screenshots)
RE_CMPS_NEG = re.compile(
    r"оплата\s*частинами|розрахунк\w*\s*\"?оплата\s*частинами\"?",
    re.IGNORECASE
)

# =========================
# Detection function
# =========================

def detect_acquiring(row):
    pp = row.get("PLATPURPOSE", "")
    cp = row.get("CONTRAGENTSANAME", "")

    pp_n = _norm(pp)
    cp_n = _norm(cp)
    text = _join_text(pp, cp)

    reasons = []

    # --- Strong / explicit ---
    if RE_TYPE_ACQ.search(text):
        reasons.append("type_acquiring")

    if RE_CP_STRONG.search(cp_n):
        reasons.append("counterparty_strong")

    # --- Medium (semantic) ---
    if RE_OPER_ACQ.search(text):
        reasons.append("operational_acq")

    if RE_REFUND.search(text):
        reasons.append("acq_refund")

    if RE_COVERAGE.search(text):
        reasons.append("cards_coverage")

    if RE_CASH.search(text):
        reasons.append("cash_epz")

    # --- CMPS as WEAK signal with guardrails ---
    if RE_CMPS.search(pp_n):
        # If negative context like "Оплата Частинами" -> do NOT classify as acquiring by CMPS
        if not RE_CMPS_NEG.search(cp_n):
            # CMPS must be "confirmed" by acquiring-ish keywords somewhere in text
            if RE_CMPS_CONFIRM.search(text):
                reasons.append("cmps_confirmed")

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(reasons),
        "acq_score": len(reasons)
    })

# =========================
# Run pipeline
# =========================

df = pd.read_excel("transactions.xlsx")  # must have PLATPURPOSE, CONTRAGENTSANAME

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
