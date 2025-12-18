(покрит\w*|вируч\w*).{0,80}(кореспонденц\w*.{0,80}\b2924\b|\b2924\b.{0,80}кореспонденц\w*)

BANK_EDRPOU = {
    "300528",  # OTP
    # сюди можна додати інші банки
}


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

    # нормалізуємо ЄДРПОУ
    result["CLIENT_EDRPOU"] = result["CONTRAGENTIDENTIFYCODE"].astype(str)

    mask_bank_edrpou = result["CLIENT_EDRPOU"].isin(BANK_EDRPOU)

    result.loc[mask_bank_edrpou, "CLIENT_EDRPOU"] = (
        result.loc[mask_bank_edrpou, "CONTRAGENTBIDENTIFYCODE"].astype(str)
    )

    result["PERIOD"] = period

    return result.drop_duplicates()
