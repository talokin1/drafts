import pandas as pd
import numpy as np

# приклад: MFO_OTP = 300528  # постав своє значення
# period = "2025-11"        # або будь-який тег періоду

def build_client_turnover(df: pd.DataFrame, MFO_OTP, period) -> pd.DataFrame:
    """
    Вхід: df транзакцій з колонками (мінімум):
      BANKAID, BANKBID,
      CONTRAGENTATID, CONTRAGENTBTID,
      CONTRAGENTATIDENTIFYCODE, CONTRAGENTBIDENTIFYCODE,
      CONTRAGENTASNAME, CONTRAGENTBSNAME,
      SUMMAEQ (або SUMMA / SUMMAEQ - див. нижче)

    Вихід: агрегація по клієнту з оборотом і банками, якими користується.
    """

    df = df.copy()

    # 0) Сума: у тебе в різних місцях буває SUMMA / SUMMAEQ
    if "SUMMAEQ" in df.columns:
        amount_col = "SUMMAEQ"
    elif "SUMMA" in df.columns:
        amount_col = "SUMMA"
    else:
        raise KeyError("Немає колонки суми: очікував SUMMAEQ або SUMMA")

    # 1) Тип транзакції (відносно OTP):
    # - DEBIT: OTP з боку A (BANKAID == MFO_OTP), а інша сторона не OTP
    # - CREDIT: OTP з боку B (BANKBID == MFO_OTP), а інша сторона не OTP
    is_debit = (df["BANKAID"] == MFO_OTP) & (df["BANKBID"] != MFO_OTP)
    is_credit = (df["BANKBID"] == MFO_OTP) & (df["BANKAID"] != MFO_OTP)

    df["TYPE"] = np.select(
        [is_debit, is_credit],
        ["DEBIT", "CREDIT"],
        default="OTHER"  # внутрішні/між OTP/некоректні
    )

    # Якщо тобі потрібні ТІЛЬКИ зовнішні (DEBIT/CREDIT) — відфільтруй:
    df = df[df["TYPE"].isin(["DEBIT", "CREDIT"])].copy()

    # 2) Логіка клієнта (A vs B) — як у твоєму старому коді:
    # DEBIT -> беремо A (CONTRAGENTA*)
    # CREDIT -> беремо B (CONTRAGENTB*)
    df["CLIENT_ID"] = np.where(df["TYPE"].eq("DEBIT"), df["CONTRAGENTAID"], df["CONTRAGENTBID"])
    df["CLIENT_IDENTIFYCODE"] = np.where(
        df["TYPE"].eq("DEBIT"),
        df["CONTRAGENTAIDENTIFYCODE"],
        df["CONTRAGENTBIDENTIFYCODE"]
    )
    df["CLIENT_NAME"] = np.where(
        df["TYPE"].eq("DEBIT"),
        df["CONTRAGENTASNAME"],
        df["CONTRAGENTBSNAME"]
    )

    # 3) BANK_USED — банк "яким користуються" = банк на іншій стороні (як у тебе було):
    # DEBIT: інша сторона = BANKBID
    # CREDIT: інша сторона = BANKAID
    df["BANK_USED"] = np.where(df["TYPE"].eq("DEBIT"), df["BANKBID"], df["BANKAID"])

    # 4) Типи/касти, щоб не було сюрпризів з float/NaN
    df["CLIENT_IDENTIFYCODE"] = df["CLIENT_IDENTIFYCODE"].astype("string")
    df["CLIENT_NAME"] = df["CLIENT_NAME"].astype("string")
    df["BANK_USED"] = df["BANK_USED"].astype("Int64").astype("string")  # щоб були коди як рядки
    df["CLIENT_ID"] = pd.to_numeric(df["CLIENT_ID"], errors="coerce")  # може бути NaN — ок

    df["ACQ_TEXT"] = np.where(
        df["TYPE"].eq("DEBIT"),
        df["PLATPURPOSE"],        # для debit беремо PLATPURPOSE
        df["PLATPURPOSE"]         # для credit теж PLATPURPOSE (за потреби можна міняти)
    )

    df["ACQ_COUNTERPARTY_NAME"] = np.where(
        df["TYPE"].eq("DEBIT"),
        df["CONTRAGENTASNAME"],
        df["CONTRAGENTBSNAME"]
    )

    def safe_mode(s: pd.Series):
        s = s.dropna().astype(str)
        if s.empty:
            return None
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]



    # 5) Агрегація обороту та банків
    # Банки: унікальні, відсортовані, через кому
    summary = (
    df.groupby(["CLIENT_IDENTIFYCODE", "TYPE"], dropna=False)
      .agg(
          n_txn=(amount_col, "count"),
          turnover=(amount_col, "sum"),
          CLIENT_NAME=("CLIENT_NAME", "first"),
          CLIENT_ID=("CLIENT_ID", "first"),
          bank_used=("BANK_USED", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),

          # 👇 ОЦЕ ГОЛОВНЕ
          ACQ_PLATPURPOSE=("ACQ_TEXT", safe_mode),
          ACQ_COUNTERPARTY_NAME=("ACQ_COUNTERPARTY_NAME", safe_mode),
      )
      .reset_index()
      .sort_values("turnover", ascending=False)
    )   


    if period is not None:
        summary["PERIOD"] = period

    return summary


df = pd.read_parquet(r"...твій_файл.parquet")
MFO_OTP = 300528  # підстав своє
summary = build_client_turnover(df, MFO_OTP=MFO_OTP, period="2025-11")
summary.head(20)
