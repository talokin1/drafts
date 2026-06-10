import re
import pandas as pd

def normalize_phone(x):
    if pd.isna(x):
        return None
    
    x = str(x).strip()
    digits = re.sub(r"\D", "", x)

    # 0804741213 -> 380804741213
    if digits.startswith("0") and len(digits) == 10:
        digits = "38" + digits

    # 8063... -> 38063...
    elif digits.startswith("80") and len(digits) == 11:
        digits = "3" + digits

    # 063... -> 38063...
    elif digits.startswith("63") and len(digits) == 9:
        digits = "380" + digits

    return digits if digits else None


def split_phones(x):
    if pd.isna(x):
        return []
    
    x = str(x).lower().strip()
    
    if x in ["no phone", "nan", "none", ""]:
        return []
    
    parts = re.split(r"[;,/|\n]+", x)
    return [p for p in parts if normalize_phone(p)]


def merge_phone_columns(row, cols=("phone", "FIRM_TELORG")):
    phones = []

    for col in cols:
        for p in split_phones(row[col]):
            norm = normalize_phone(p)
            if norm and norm not in phones:
                phones.append(norm)

    return "; ".join(phones)


df["phones_merged"] = df.apply(
    merge_phone_columns,
    axis=1,
    cols=("phone", "FIRM_TELORG")
)

