import pandas as pd

def normalize_phone(phone):
    if pd.isna(phone):
        return None
    
    phone = str(phone)
    
    # пропускаємо замасковані
    if "x" in phone.lower():
        return None
    
    # залишаємо тільки цифри
    digits = ''.join(filter(str.isdigit, phone))
    
    # має бути 10 цифр (0660120498)
    if len(digits) == 10 and digits.startswith("0"):
        digits = "38" + digits
    
    return digits

autoria["MOBILEPHONE"] = autoria["Номер телефону"].apply(normalize_phone)