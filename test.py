def detect_credit_acquiring(df, period):
    df = df.copy()

    # 1. Кредитова операція: кошти зайшли в OTP
    is_credit = df["BANKBID"] == MFO_OTP

    # 2. Відправник НЕ OTP (тобто зовнішнє зарахування)
    is_not_otp_sender = df["BANKAID"] != MFO_OTP

    # 3. Еквайринг по тексту / бух. формулюваннях
    is_acq_text = is_acquiring_only(df["PLATPURPOSE"])

    # 4. Фінальний фільтр (БЕЗ перевірки ЄДРПОУ)
    result = df[
        is_credit &
        is_not_otp_sender &
        is_acq_text
    ].copy()

    result["PERIOD"] = period

    return result.drop_duplicates()
