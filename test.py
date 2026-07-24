clients["MOBILEPHONE"] = clients["MOBILEPHONE"].astype(str).str.replace("+", "", regex=False)
phone["MOBILEPHONE"] = phone["MOBILEPHONE"].astype(str).str.replace("+", "", regex=False)

clients["phone_exists"] = clients["MOBILEPHONE"].isin(phone["MOBILEPHONE"])

clients["phone_exists"].value_counts()