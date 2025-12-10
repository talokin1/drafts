debit = merged_[merged_["TYPE"] == "DEBIT_SELF_ACQ"][["SEGMENT", "CLIENT_IDENTIFYCODE"]]
credit = merged_[merged_["TYPE"] == "CREDIT_SELF_ACQ"][["SEGMENT", "CLIENT_IDENTIFYCODE"]]

# групуємо по сегменту
debit_grp = debit.groupby("SEGMENT")["CLIENT_IDENTIFYCODE"].apply(set)
credit_grp = credit.groupby("SEGMENT")["CLIENT_IDENTIFYCODE"].apply(set)

# перетин клієнтів у кожному сегменті
duplicates_per_segment = {
    seg: debit_grp.get(seg, set()) & credit_grp.get(seg, set())
    for seg in merged_["SEGMENT"].unique()
}

dup_rows = []

for seg in duplicates_per_segment:
    dup_rows.append({
        "SEGMENT": seg,
        "clients_in_both_types": len(duplicates_per_segment[seg]),
        "clients_only_debit": len(debit_grp.get(seg, set()) - credit_grp.get(seg, set())),
        "clients_only_credit": len(credit_grp.get(seg, set()) - debit_grp.get(seg, set())),
        "total_unique_clients": len(debit_grp.get(seg, set()) | credit_grp.get(seg, set()))
    })

duplicates_summary = pd.DataFrame(dup_rows)
duplicates_summary


duplicate_clients_table = []

for seg in duplicates_per_segment:
    for client in duplicates_per_segment[seg]:
        duplicate_clients_table.append({
            "SEGMENT": seg,
            "CLIENT_IDENTIFYCODE": client
        })

duplicate_clients_table = pd.DataFrame(duplicate_clients_table)
duplicate_clients_table


segment_stats["clients_in_both_types"] = segment_stats["SEGMENT"].map(
    lambda seg: len(duplicates_per_segment.get(seg, set()))
)

segment_stats
