import pandas as pd
import numpy as np
import re

def normalize_kved(val):
    if pd.isna(val):
        return np.nan

    s = str(val).strip()

    # 1. заміна коми на крапку
    s = s.replace(',', '.')

    # 2. залишаємо тільки цифри і одну крапку
    s = re.sub(r'[^0-9.]', '', s)

    if '.' not in s:
        # якщо раптом без підкласу → вважаємо .00
        return f"{int(s)}.00"

    main, sub = s.split('.', 1)

    # 3. прибираємо leading zeros
    main = str(int(main)) if main else "0"

    # 4. підклас → рівно 2 цифри
    sub = sub[:2].ljust(2, '0')

    return f"{main}.{sub}"
temp["FIRM_KVED"] = temp["FIRM_KVED"].apply(normalize_kved)
