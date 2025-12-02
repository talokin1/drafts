bank_names = (
    mfos.assign(BANKID = mfos["BANKID"].astype(str).str.replace(r"\D", "", regex=True))
        .set_index("BANKID")["NAME"]
        .to_dict()
)

def convert_bank_used_to_names(value):
    s = str(value)
    parts = [x.strip() for x in s.split(",")]
    names = [bank_names.get(code, code) for code in parts]
    return ", ".join(names)

summary["bank_used"] = summary["bank_used"].apply(convert_bank_used_to_names)
