import re
from collections import defaultdict

import pandas as pd
import pdfplumber

KVED_COLS = ["KVED", "KVED_2", "KVED_3", "KVED_4"]

OLD_RE = re.compile(r"^\d{2}\.\d{2}\.\d$")   # 01.11.0
NEW_RE = re.compile(r"^\d{2}\.\d{2}$")       # 01.11

OLD_ANY_RE = re.compile(r"\b\d{2}\.\d{2}\.\d\b")
NEW_ANY_RE = re.compile(r"\b\d{2}\.\d{2}\b")


def _clean_str(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    # NBSP -> space, trim, collapse spaces
    x = x.replace("\u00A0", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_old_kved(x: str) -> str:
    """
    Нормалізує старий код до виду XX.XX.X
    Підхоплює часті криві варіанти, але не вигадує.
    """
    s = _clean_str(x)
    if not s:
        return s

    # Візьмемо перший токен, якщо в клітинці "01.11.0 01.11"
    s = s.split()[0]

    # якщо вже норм
    if OLD_RE.match(s):
        return s

    # інколи буває "84.10.0" (це вже норм), або "84.1.0" (не норм)
    # спробуємо привести "84.1.0" -> "84.01.0"
    m = re.fullmatch(r"(\d{2})\.(\d{1})\.(\d)", s)
    if m:
        return f"{m.group(1)}.0{m.group(2)}.{m.group(3)}"

    return s


def normalize_new_kved(x: str) -> str:
    s = _clean_str(x)
    if not s:
        return s
    s = s.split()[0]
    if NEW_RE.match(s):
        return s
    # інколи може вилізти "01.11." або "01.11," — підчистимо
    s = re.sub(r"[^\d\.]", "", s)
    return s


def parse_kved_mapping_from_pdf_tables(pdf_path: str) -> dict:
    """
    Парсимо PDF як таблиці.
    Для кожного рядка беремо перший OLD-код і перший NEW-код (2010) у цьому ж рядку.
    Якщо в рядку кілька NEW — збережемо їх всі, але конвертація візьме перший.
    """
    mapping = defaultdict(list)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                for row in table:
                    if not row:
                        continue
                    cells = [_clean_str(c) for c in row if c is not None]
                    if not cells:
                        continue

                    row_text = " | ".join(cells)

                    old_codes = OLD_ANY_RE.findall(row_text)
                    if not old_codes:
                        continue

                    # Важливо: NEW коди беремо з рядка, але відфільтруємо ті, що співпадають з OLD без останнього ".0"
                    new_codes = NEW_ANY_RE.findall(row_text)
                    if not new_codes:
                        continue

                    old = normalize_old_kved(old_codes[0])

                    # приберемо дублі й нормалізуємо
                    cleaned_new = []
                    for n in new_codes:
                        nn = normalize_new_kved(n)
                        if NEW_RE.match(nn) and nn not in cleaned_new:
                            cleaned_new.append(nn)

                    if not cleaned_new:
                        continue

                    # додамо всі варіанти (порядок як у рядку)
                    for nn in cleaned_new:
                        if nn not in mapping[old]:
                            mapping[old].append(nn)

    return dict(mapping)


def is_old_kved(x) -> bool:
    if not isinstance(x, str):
        return False
    return bool(OLD_RE.match(normalize_old_kved(x)))


def convert_one(code, mapping):
    if not isinstance(code, str):
        return code, False

    old = normalize_old_kved(code)
    if OLD_RE.match(old) and old in mapping and mapping[old]:
        return mapping[old][0], True
    return code, False


def convert_df(df: pd.DataFrame, mapping: dict):
    df = df.copy()

    old_before = 0
    old_after = 0
    converted = 0
    unmatched_old = defaultdict(int)

    for col in KVED_COLS:
        if col not in df.columns:
            continue

        out = []
        for v in df[col].astype("string").fillna(""):
            v = _clean_str(v)
            if is_old_kved(v):
                old_before += 1
                norm_old = normalize_old_kved(v)
                if norm_old not in mapping:
                    unmatched_old[norm_old] += 1

            nv, ok = convert_one(v, mapping)
            if ok:
                converted += 1

            if is_old_kved(_clean_str(nv)):
                old_after += 1

            out.append(nv if nv != "" else pd.NA)

        df.loc[:, col] = out

    return df, old_before, converted, old_after, unmatched_old


def main():
    PDF_PATH = r"C:\path\to\KVED_2005_to_2010.pdf"
    INPUT_CSV = r"C:\path\to\data.csv"
    OUTPUT_CSV = r"C:\path\to\data_kved2010.csv"

    print("Parsing PDF (tables)...")
    mapping = parse_kved_mapping_from_pdf_tables(PDF_PATH)
    print(f"Mapping size: {len(mapping)}")

    print("Loading data...")
    df = pd.read_csv(INPUT_CSV, dtype={c: "string" for c in KVED_COLS}, keep_default_na=True)

    print("Converting KVEDs...")
    df2, old_before, converted, old_after, unmatched_old = convert_df(df, mapping)

    print("Saving result...")
    df2.to_csv(OUTPUT_CSV, index=False)

    print("---- KVED CONVERSION STATS ----")
    print(f"Old KVEDs before : {old_before}")
    print(f"Converted        : {converted}")
    print(f"Old KVEDs after  : {old_after}")
    print("--------------------------------")

    if old_before > 0:
        top_unmatched = sorted(unmatched_old.items(), key=lambda x: x[1], reverse=True)[:20]
        if top_unmatched:
            print("Top unmatched OLD KVEDs (not found in mapping):")
            for k, cnt in top_unmatched:
                print(f"  {k}: {cnt}")


if __name__ == "__main__":
    main()
