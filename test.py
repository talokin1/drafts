import pandas as pd
from collections import Counter
import re

# -------------------------
# 1. Load & prep
# -------------------------

df = pd.read_excel("acquiring_table.xlsx")

df = df.rename(columns={
    "ОзнАка екв": "is_acq",
    "PLATPURPOSE": "text"
})

df["is_acq"] = (df["is_acq"].str.lower() == "екв")
df["text"] = df["text"].fillna("").str.lower()

def normalize(s):
    s = re.sub(r"\s+", " ", s)
    return s.strip()

df["text"] = df["text"].apply(normalize)

df_acq = df[df["is_acq"]]
df_non = df[~df["is_acq"]]

# -------------------------
# 2. Substring extraction
# -------------------------

def substrings(s, min_len=6, max_len=20):
    subs = set()
    L = len(s)
    for l in range(min_len, max_len + 1):
        for i in range(0, L - l + 1):
            subs.add(s[i:i+l])
    return subs

# -------------------------
# 3. Count presence (not frequency)
# -------------------------

acq_counter = Counter()
non_counter = Counter()

for text in df_acq["text"]:
    for sub in substrings(text):
        acq_counter[sub] += 1

for text in df_non["text"]:
    for sub in substrings(text):
        non_counter[sub] += 1

# -------------------------
# 4. Build stats table
# -------------------------

records = []

N_acq = len(df_acq)
N_non = len(df_non)

for sub, cnt_acq in acq_counter.items():
    cnt_non = non_counter.get(sub, 0)

    coverage = cnt_acq / N_acq
    leakage = cnt_non / max(1, N_non)

    records.append({
        "pattern": sub,
        "len": len(sub),
        "acq_coverage": coverage,
        "non_coverage": leakage,
        "lift": (coverage + 1e-6) / (leakage + 1e-6)
    })

patterns = pd.DataFrame(records)

# -------------------------
# 5. Strong patterns
# -------------------------

good = patterns[
    (patterns["acq_coverage"] > 0.3) &      # ≥30% еквайрингу
    (patterns["non_coverage"] < 0.05) &     # майже не поза екв
    (patterns["len"] <= 25)
].sort_values(
    ["acq_coverage", "len"],
    ascending=[False, True]
)

good.head(30)
