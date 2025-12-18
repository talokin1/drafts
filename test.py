def detect_credit_acquiring(df, period):
    df = df.copy()

    is_credit = df["BANKBID"] == MFO_OTP
    is_not_otp_sender = df["BANKAID"] != MFO_OTP
    is_acq_text = is_acquiring_only(df["PLATPURPOSE"])

    result = df[
        is_credit &
        is_not_otp_sender &
        is_acq_text
    ].copy()

    # базово беремо A
    result["CLIENT_EDRPOU"] = result["CONTRAGENTAIDENTIFYCODE"].astype(str)

    # якщо A — банк, то клієнт у B
    mask_bank_edrpou = result["CLIENT_EDRPOU"].isin(BANK_EDRPOU)

    result.loc[mask_bank_edrpou, "CLIENT_EDRPOU"] = (
        result.loc[mask_bank_edrpou, "CONTRAGENTBIDENTIFYCODE"].astype(str)
    )

    result["PERIOD"] = period

    return result.drop_duplicates()
