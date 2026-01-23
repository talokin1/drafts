import json
import pandas as pd
import re

# =========================
# CONFIG
# =========================
JSON_PATH = r"C:\path\to\kved_2010_structure.json"
INPUT_CSV = r"C:\path\to\data_kved2010.csv"
OUTPUT_CSV = r"C:\path\to\data_with_kved_structure.csv"

KVED_COLS = ["KVED", "KVED_2", "KVED_3", "KVED_4"]

KVED_RE = re.compile(r"^\d{2}\.\d{2}$")


# =========================
# LOAD JSON
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    KVED_DICT = json.load(f)


# =========================
# HELPERS
# =========================
def clean_kved(x):
    if not isinstance(x, str):
        return None
    x = x.strip()
    if KVED_RE.match(x):
        return x
    return None


def extract_main_kved(row):
    for col in KVED_COLS:
        k = clean_kved(row.get(col))
        if k and k in KVED_DICT:
            return k
    return None


def build_kved_structure(row):
    kved = extract_main_kved(row)
    if not kved:
        return pd.Series(
            {
                "KVED_MAIN": pd.NA,
                "KVED_CLASS": pd.NA,
                "KVED_SECTION": pd.NA,
                "KVED_SECTION_CODE": pd.NA,
            }
        )

    meta = KVED_DICT[kved]

    return pd.Series(
        {
            "KVED_MAIN": kved,
            "KVED_CLASS": meta.get("class_code"),
            "KVED_SECTION": meta.get("section_name"),
            "KVED_SECTION_CODE": meta.get("section_code"),
        }
    )


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    print("Building KVED structure...")
    kved_struct = df.apply(build_kved_structure, axis=1)

    df = pd.concat([df, kved_struct], axis=1)

    print("Saving result...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
