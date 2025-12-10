debit_clients = set(summary_[summary_["TYPE"]=="DEBIT_SELF_ACQ"]["CLIENT_IDENTIFYCODE"])
credit_clients = set(summary_[summary_["TYPE"]=="CREDIT_SELF_ACQ"]["CLIENT_IDENTIFYCODE"])

duplicate_clients = debit_clients & credit_clients

dups = summary_[summary_["CLIENT_IDENTIFYCODE"].isin(duplicate_clients)].copy()
dups = dups.sort_values(["CLIENT_IDENTIFYCODE", "TYPE"])
dups["HIGHLIGHT"] = True

summary_["HIGHLIGHT"] = summary_["CLIENT_IDENTIFYCODE"].isin(duplicate_clients)

summary_sorted = summary_.sort_values(
    ["HIGHLIGHT", "CLIENT_IDENTIFYCODE", "TYPE"],
    ascending=[False, True, True]   # HIGHLIGHT=True → згори
)

def highlight_dups(row):
    if row["HIGHLIGHT"]:
        return ['background-color: lightgreen'] * len(row)
    return [''] * len(row)

styled = summary_sorted.style.apply(highlight_dups, axis=1)
styled

import pandas as pd

with pd.ExcelWriter("summary_with_highlight.xlsx", engine="xlsxwriter") as writer:
    summary_sorted.to_excel(writer, index=False, sheet_name="Summary")

    workbook = writer.book
    worksheet = writer.sheets["Summary"]

    green = workbook.add_format({"bg_color": "#CCFFCC"})  # light green

    highlight_col = summary_sorted.columns.get_loc("HIGHLIGHT")

    for row_idx, val in enumerate(summary_sorted["HIGHLIGHT"], start=1):
        if val:
            worksheet.set_row(row_idx, cell_format=green)
