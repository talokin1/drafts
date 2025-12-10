df["TYPE_NEW"] = np.where(
    df["CONTRAGENTAIDENTIFYCODE"].astype(str)
        == df["CONTRAGENTBIDENTIFYCODE"].astype(str),
    "CREDIT_SELF_ACQ",
    "DEBIT_SELF_ACQ"
)
