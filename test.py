import pdfplumber
import pandas as pd
import re
from collections import defaultdict

# -------------------------------------------------
# REGEX
# -------------------------------------------------
OLD_KVED_RE = re.compile(r"\b\d{2}\.\d{2}\.\d\b")   # 01.11.0
NEW_KVED_RE = re.compile(r"\b\d{2}\.\d{2}\b")       # 01.11


# -------------------------------------------------
# 1. PARSE PDF -> MAPPING
# -------------------------------------------------
def parse_kved_mapping_from_pdf(pdf_path: str) -> dict:
    """
    Returns:
    {
        '01.11.0': ['01.11'],
        '28.52.0': ['25.62', '33.11'],
        ...
    }
    """
    mapping = defaultdict(list)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                old_codes = OLD_KVED_RE.findall(line)
                new_codes = NEW_KVED_RE.findall(line)

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
        return code

    code = code.strip()

    if OLD_KVED_RE.fullmatch(code):
        if code in mapping and mapping[code]:
            return mapping[code][0]   # беремо перший
    return code


# -------------------------------------------------
# 3. APPLY TO DATAFRAME
# -------------------------------------------------
def convert_kved_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    kved_cols = ["KVED", "KVED_2", "KVED_3", "KVED_4"]

    for col in kved_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: convert_kved_2005_to_2010(x, mapping))

    return df


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
    df = convert_kved_columns(df, kved_mapping)

    print("Saving result...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()


import re

OLD_RE = re.compile(r"^\d{2}\.\d{2}\.\d$")

(df[["KVED", "KVED_2", "KVED_3", "KVED_4"]]
 .applymap(lambda x: isinstance(x, str) and OLD_RE.match(x))
 .sum())
