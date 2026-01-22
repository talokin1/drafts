import re
import unicodedata

_APOS_RE = re.compile(r"[’ʻʼ`´]")
_DASH_RE = re.compile(r"[‐-–—−]")

def norm_ua(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()

    # уніфікуємо апострофи
    s = _APOS_RE.sub("'", s)

    # уніфікуємо тире/дефіси
    s = _DASH_RE.sub("-", s)

    # прибираємо лапки типу «»
    s = re.sub(r"[«»\"“”]", " ", s)

    # схлопуємо пробіли
    s = re.sub(r"\s+", " ", s).strip()
    return s

opf_ref["OPF_NAME_NORM"] = opf_ref["OPF_NAME"].map(norm_ua)
df["FULL_NAME_NORM"] = df["Повна назва"].map(norm_ua)

opf_ref = opf_ref.sort_values("OPF_NAME_NORM", key=lambda s: s.str.len(), ascending=False)

a = norm_ua("Адвокатське об’єднання")   # з excel (часто тут ’)
b = norm_ua("АДВОКАТСЬКЕ ОБ'ЄДНАННЯ")   # з FULL_NAME
print(a, b, a == b)
