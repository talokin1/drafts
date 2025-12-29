def normalize_kved(val):
    if pd.isna(val):
        return np.nan

    s = str(val).strip()
    s = s.replace(',', '.')

    # тільки цифри і крапка
    s = re.sub(r'[^0-9.]', '', s)

    # ВАЛІДАЦІЯ
    if not re.fullmatch(r'\d{1,2}(\.\d{1,2})?', s):
        return np.nan   # або "INVALID_KVED"

    if '.' not in s:
        return f"{int(s)}.00"

    main, sub = s.split('.', 1)
    main = str(int(main))
    sub = sub.ljust(2, '0')

    return f"{main}.{sub}"

temp["FIRM_KVED_CLEAN"] = temp["FIRM_KVED"].apply(normalize_kved)
