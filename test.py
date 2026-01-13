def normalize_ua(text: str) -> str:
    table = str.maketrans({
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
        "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р",
        "T": "Т", "X": "Х",
        "a": "а", "b": "в", "c": "с", "e": "е", "h": "н",
        "i": "і", "k": "к", "m": "м", "o": "о", "p": "р",
        "t": "т", "x": "х",
    })
    return text.translate(table)

test_norm = normalize_ua(test)
re.search(r"відшк", test_norm, re.IGNORECASE)
