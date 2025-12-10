# розбиваємо на два набори
debit_clients = (
    merged_[merged_["TYPE"] == "DEBIT_SELF_ACQ"]
    .groupby("SEGMENT")["CLIENT_IDENTIFYCODE"]
    .unique()
)

credit_clients = (
    merged_[merged_["TYPE"] == "CREDIT_SELF_ACQ"]
    .groupby("SEGMENT")["CLIENT_IDENTIFYCODE"]
    .unique()
)

# перетин клієнтів
intersection = {
    seg: set(debit_clients.get(seg, [])) & set(credit_clients.get(seg, []))
    for seg in merged_["SEGMENT"].unique()
}

intersection

intersection_count = {
    seg: len(intersection[seg])
    for seg in intersection
}

intersection_count

rows = []
for seg in intersection:
    rows.append({
        "SEGMENT": seg,
        "clients_in_both": len(intersection[seg]),
        "clients_only_debit": len(set(debit_clients.get(seg, [])) - set(credit_clients.get(seg, []))),
        "clients_only_credit": len(set(credit_clients.get(seg, [])) - set(debit_clients.get(seg, []))),
        "total_unique_clients": len(set(debit_clients.get(seg, [])) | set(credit_clients.get(seg, [])))
    })

intersection_df = pd.DataFrame(rows)
intersection_df
