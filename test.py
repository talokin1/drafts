import re
from collections import defaultdict

import pandas as pd
import pdfplumber


# =========================
# CONFIG
# =========================
PDF_PATH = r"C:\path\to\KVED_2005_to_2010.pdf"
INPUT_CSV = r"C:\path\to\data.csv"
OUTPUT_CSV = r"C:\path\to\data_kved2010.csv"

KVED_COLS = ["KVED", "KVED_2", "KVED_3", "KVED_4"]


# =========================
# REGEX
# =========================
OLD_RE = re.compile(r"^\d{2}\.\d{2}\.\d$")
NEW_RE = re.compile(r"^\d{2}\.\d{2}$")

OLD_ANY_RE = re.compile(r"\b\d{2}\.\d{2}\.\d\b")
NEW_ANY_RE = re.compile(r"\b\d{2}\.\d{2}\b")


# =========================
# HELPERS
# =========================
def clean_str(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.replace("\u00A0", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_old_kved(x: str) -> str:
    x = clean_str(x)
    if not x:
        return x

    x = x.split()[0]

    if OLD_RE.match(x):
        return x

    m = re.fullmatch(r"(\d{2})\.(\d{1})\.(\d)", x)
    if m:
        return f"{m.group(1)}.0{m.group(2)}.{m.group(3)}"

    return x


def collapse_old_kved_to_zero(x: str) -> str:
    """
    15.33.2 -> 15.33.0
    """
    m = re.fullmatch(r"(\d{2}\.\d{2})\.\d", x)
    if m:
        return f"{m.group(1)}.0"
    return x


# =========================
# PDF → MAPPING
# =========================
def parse_kved_mapping_from_pdf(pdf_path: str) -> dict:
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

                    cells = [clean_str(c) for c in row if c]
                    if not cells:
                        continue

                    row_text = " | ".join(cells)

                    old_codes = OLD_ANY_RE.findall(row_text)
                    if not old_codes:
                        continue

                    new_codes = NEW_ANY_RE.findall(row_text)
                    if not new_codes:
                        continue

                    old = normalize_old_kved(old_codes[0])

                    cleaned_new = []
                    for n in new_codes:
                        n = clean_str(n)
                        if NEW_RE.match(n) and n not in cleaned_new:
                            cleaned_new.append(n)

                    for n in cleaned_new:
                        if n not in mapping[old]:
                            mapping[old].append(n)

    return dict(mapping)


# =========================
# CONVERSION LOGIC
# =========================
def convert_one(code, mapping):
    if not isinstance(code, str):
        return code, False

    code = normalize_old_kved(code)

    # 1️⃣ пряме співпадіння
    if OLD_RE.match(code) and code in mapping and mapping[code]:
        return mapping[code][0], True

    # 2️⃣ згортання підкласу *.X → *.0
    collapsed = collapse_old_kved_to_zero(code)
    if collapsed != code:
        if collapsed in mapping and mapping[collapsed]:
            return mapping[collapsed][0], True

    return code, False


# =========================
# MAIN PROCESS
# =========================
def main():
    print("Parsing PDF...")
    mapping = parse_kved_mapping_from_pdf(PDF_PATH)
    print(f"Mapping size: {len(mapping)}")

    print("Loading CSV...")
    df = pd.read_csv(
        INPUT_CSV,
        dtype={c: "string" for c in KVED_COLS},
        keep_default_na=True
    )
    df = df.copy()

    old_before = 0
    converted = 0
    old_after = 0

    print("Converting KVEDs...")
    for col in KVED_COLS:
        if col not in df.columns:
            continue

        new_vals = []
        for v in df[col].fillna(""):
            v = clean_str(v)

            if OLD_RE.match(normalize_old_kved(v)):
                old_before += 1

            nv, ok = convert_one(v, mapping)
            if ok:
                converted += 1

            if OLD_RE.match(normalize_old_kved(nv)):
                old_after += 1

            new_vals.append(nv if nv else pd.NA)

        df.loc[:, col] = new_vals

    print("Saving result...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("---- KVED CONVERSION STATS ----")
    print(f"Old KVEDs before : {old_before}")
    print(f"Converted        : {converted}")
    print(f"Old KVEDs after  : {old_after}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
