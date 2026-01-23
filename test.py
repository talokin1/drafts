def has_old_kved_in_row(row) -> bool:
    for col in KVED_COLS:
        val = row.get(col)
        if isinstance(val, str) and OLD_RE.match(normalize_old_kved(val)):
            return True
    return False


print("Filtering rows with remaining old KVEDs...")

before_rows = len(df)

mask_old = df.apply(has_old_kved_in_row, axis=1)
df = df.loc[~mask_old].copy()

after_rows = len(df)

print("---- FILTER STATS ----")
print(f"Rows before filtering : {before_rows}")
print(f"Rows removed          : {before_rows - after_rows}")
print(f"Rows after filtering  : {after_rows}")
print("----------------------")
