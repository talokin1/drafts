RE_COMMISSION = re.compile(
    r"(cmps|cmp|коміс|ком\.?|komis|commission).*?(\d+[.,]?\d*)",
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
    r"відшк\w*.*?екв",
    re.IGNORECASE
)

RE_TYPE_ACQ = re.compile(
    r"type\s*acquir|liqpay|split\s+id",
    re.IGNORECASE
)

RE_COUNTERPARTY = re.compile(
    r"еквайр|acquir|liqpay",
    re.IGNORECASE
)



def detect_acquiring(row):
    text = f"{row.get('PLATPURPOSE', '')} {row.get('CONTRAGENTSANAME', '')}".lower()

    reasons = []

    if RE_TYPE_ACQ.search(text):
        reasons.append("type_acquiring")
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
    if RE_COUNTERPARTY.search(text):
        reasons.append("counterparty_name")

    return pd.Series({
        "is_acquiring": len(reasons) > 0,
        "acq_reason": "|".join(reasons),
        "acq_score": len(reasons)
    })


# === LOAD DATA ===
df = pd.read_excel("transactions.xlsx")  
# обовʼязково мають бути колонки:
# PLATPURPOSE, CONTRAGENTSANAME

# === APPLY DETECTOR ===
detected = df.apply(detect_acquiring, axis=1)

df = pd.concat([df, detected], axis=1)


# Загальна статистика
print("Знайдено еквайрингу:", df["is_acquiring"].sum())
print("Відсоток:", round(df["is_acquiring"].mean() * 100, 2), "%")

# По причинах
print("\nBreakdown по правилах:")
print(
    df[df["is_acquiring"]]
    .groupby("acq_reason")
    .size()
    .sort_values(ascending=False)
)


df.to_excel(
    "transactions_with_acquiring_flag.xlsx",
    index=False
)
