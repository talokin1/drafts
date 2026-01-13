pp = trxs["PLATPURPOSE"].fillna("").str.lower()
cp = trxs["CONTRAGENTASNAME"].fillna("").str.lower()

m_cmps_merchant = pp.str.contains(RE_CMPS_MERCHANT)
m_commission    = pp.str.contains(RE_COMMISSION)
m_installments  = pp.str.contains(RE_INSTALLMENTS)
m_refund_strong = pp.apply(normalize_ua).str.contains(RE_REFUND_STRONG)

m_acq_settle    = pp.str.contains(RE_ACQ_SETTLEMENT)
m_refund_cash   = pp.str.contains(RE_REFUND_CASH)
m_merchant_cash = pp.str.contains(RE_MERCHANT_CASH)

m_household     = pp.str.contains(RE_HOUSEHOLD)
m_bank_like     = pp.str.contains(RE_BANK_LIKE)
m_oper_acq      = pp.str.contains(RE_OPER_ACQ)

m_part_pay      = pp.str.contains(RE_PART_PAY)

m_cp_inet       = cp.str.contains(RE_INTERNET_ACQ_CP)
m_cp_keyword    = cp.str.contains(RE_COUNTERPARTY)

m_type_acq      = pp.str.contains(RE_TYPE_ACQ)
m_oper_acq      = pp.str.contains(RE_OPER_ACQ)
m_coverage      = pp.str.contains(RE_COVERAGE)

df["is_acquiring"] = False
df["acq_reason"] = ""
df["acq_score"] = 0

mask = m_cmps_merchant
df.loc[mask, ["is_acquiring", "acq_reason", "acq_score"]] = [True, "cmps_merchant_commission", 1]

mask = m_acq_settle & ~df["is_acquiring"]
df.loc[mask, ["is_acquiring", "acq_reason", "acq_score"]] = [True, "acq_settlement", 1]

mask = m_refund_strong & ~df["is_acquiring"]
df.loc[mask, ["is_acquiring", "acq_reason", "acq_score"]] = [True, "refund", 1]

mask = m_refund_cash & m_cmps_merchant & ~df["is_acquiring"]
df.loc[mask, ["is_acquiring", "acq_reason", "acq_score"]] = [True, "refund_merchant_cash", 2]

mask = m_merchant_cash & ~df["is_acquiring"]
df.loc[mask, ["is_acquiring", "acq_reason", "acq_score"]] = [True, "merchant_cash_withdrawal", 1]


mask = m_part_pay
df.loc[mask, ["is_acquiring", "acq_reason", "acq_score"]] = [False, "", 0]

mask = m_household & m_bank_like & ~m_oper_acq
df.loc[mask, ["is_acquiring", "acq_reason", "acq_score"]] = [False, "", 0]


cp_score = m_cp_inet.astype(int) + m_cp_keyword.astype(int)
mask = (cp_score > 0) & ~df["is_acquiring"]

df.loc[mask, "is_acquiring"] = True
df.loc[mask, "acq_reason"] = (
    np.where(m_cp_inet, "cp_internet_acquiring", "")
    + np.where(m_cp_keyword, "|cp_acquiring_keyword", "")
).str.strip("|")
df.loc[mask, "acq_score"] = cp_score


pp_score = (
    m_type_acq.astype(int)
    + m_oper_acq.astype(int)
    + ((m_commission & (m_oper_acq | m_type_acq)).astype(int))
    + m_coverage.astype(int)
)

mask = (pp_score > 0) & ~df["is_acquiring"]
df.loc[mask, "is_acquiring"] = True
df.loc[mask, "acq_score"] = pp_score
