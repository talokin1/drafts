import pandas as pd
import re
import ast

def extract_phone(val):

    if val == 0 or pd.isna(val):
        return None

    try:
        # перетворюємо рядок у dict
        if isinstance(val, str) and val.startswith("{"):
            val = ast.literal_eval(val)

        phone = val.get("phone")

        if phone is None:
            return None

        # пропускаємо masked
        if "x" in phone.lower():
            return None

        # залишаємо тільки цифри
        digits = re.sub(r"\D", "", phone)

        if len(digits) == 10 and digits.startswith("0"):
            digits = "38" + digits

        return digits

    except:
        return None


autoria["MOBILEPHONE"] = autoria["userPhoneData"].apply(extract_phone)
autoria = autoria.dropna(subset=["MOBILEPHONE"])