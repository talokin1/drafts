debit_clients = set(merged_[merged_["TYPE"] == "DEBIT_SELF_ACQ"]["CLIENT_IDENTIFYCODE"])
credit_clients = set(merged_[merged_["TYPE"] == "CREDIT_SELF_ACQ"]["CLIENT_IDENTIFYCODE"])
duplicate_ids_df = pd.DataFrame({"CLIENT_IDENTIFYCODE": list(duplicate_ids)})
duplicate_ids_df


dups_per_segment = (
    merged_[merged_["CLIENT_IDENTIFYCODE"].isin(duplicate_ids)]
    .groupby("SEGMENT")["CLIENT_IDENTIFYCODE"]
    .unique()
    .reset_index()
)

dups_per_segment


duplicates_full = merged_[merged_["CLIENT_IDENTIFYCODE"].isin(duplicate_ids)]
duplicates_full.sort_values(["CLIENT_IDENTIFYCODE", "TYPE"])

