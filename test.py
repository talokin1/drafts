import pdfplumber
import pandas as pd
import re
from collections import defaultdict

# -------------------------------------------------
# REGEX
# -------------------------------------------------
OLD_KVED_RE = re.compile(r"^\d{2}\.\d{2}\.\d$")   # 01.11.0
NEW_KVED_RE = re.compile(r"\b\d{2}\.\d{2}\b")     # 01.11


# -------------------------------------------------
# 1. PARSE PDF -> MAPPING
# -------------------------------------------------
def parse_kved_mapping_from_pdf(pdf_path: str) -> dict:
    mapping = defaultdict(list)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                old_codes = re.findall(r"\b\d{2}\.\d{2}\.\d\b", line)
                new_codes = re.findall(r"\b\d{2}\.\d{2}\b", line)

                if not old_codes or not new_codes:
                    continue

                for old in old_codes:
                    for new in new_codes:
                        if new not in mapping[old]:
                            mapping[old].append(new)

    return dict(mapping)


# -------------------------------------------------
# 2. CONVERT SINGLE KVED
# -------------------------------------------------
def convert_kved_2005_to_2010(code, mapping):
    if not isinstance(code, str):
        return code, False

    code = code.strip()

    if OLD_KVED_RE.match(code):
        if code in mapping and mapping[code]:
            return mapping[code][0], True

    return code, False


# -------------------------------------------------
# 3. APPLY + TRACK STATS
# -------------------------------------------------
def convert_kved_columns_with_stats(df: pd.DataFrame, mapping: dict):
    kved_cols = ["KVED", "KVED_2", "KVED_3", "KVED_4"]

    old_before = 0
    old_after = 0
    converted = 0

    for col in kved_cols:
        if col not in df.columns:
            continue

        new_col = []
        for val in df[col]:
            if isinstance(val, str) and OLD_KVED_RE.match(val.strip()):
                old_before += 1

            new_val, was_converted = convert_kved_2005_to_2010(val, mapping)

            if was_converted:
                converted += 1

            if isinstance(new_val, str) and OLD_KVED_RE.match(new_val.strip()):
                old_after += 1

            new_col.append(new_val)

        df[col] = new_col

    return df, old_before, converted, old_after


# -------------------------------------------------
# 4. MAIN
# -------------------------------------------------
def main():
    PDF_PATH = r"C:\path\to\KVED_2005_to_2010.pdf"
    INPUT_CSV = r"C:\path\to\data.csv"
    OUTPUT_CSV = r"C:\path\to\data_kved2010.csv"

    print("Parsing PDF...")
    kved_mapping = parse_kved_mapping_from_pdf(PDF_PATH)
    print(f"Mapping size: {len(kved_mapping)}")

    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)

    print("Converting KVEDs...")
    df, old_before, converted, old_after = convert_kved_columns_with_stats(df, kved_mapping)

    print("Saving result...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("---- KVED CONVERSION STATS ----")
    print(f"Old KVEDs before : {old_before}")
    print(f"Converted        : {converted}")
    print(f"Old KVEDs after  : {old_after}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
