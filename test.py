artifact = {
    # ---------- Primary ----------
    "primary": {
        "model": lgc_primary,
        "features": features_primary,
        "threshold": 0.5
    },

    "income_bins": {
        "model": multiclass_model,
        "features": features_multiclass,
        "label_encoder": label_encoder,
        "bin_edges": bin_edges,
        "bin_labels": bin_labels
    }
}

import pickle

with open("corp_liabilities_pipeline.pkl", "wb") as f:
    pickle.dump(artifact, f)





with open("corp_liabilities_pipeline.pkl", "rb") as f:
    artifact = pickle.load(f)
X_primary = main_dataset[artifact["primary"]["features"]]

main_dataset["PRIMARY_SCORE"] = (
    artifact["primary"]["model"]
    .predict_proba(X_primary)[:, 1]
)

X_bins = main_dataset[artifact["income_bins"]["features"]]

proba_bins = artifact["income_bins"]["model"].predict_proba(X_bins)

bins_df = pd.DataFrame(
    proba_bins,
    columns=artifact["income_bins"]["label_encoder"].classes_
)

main_dataset = pd.concat([main_dataset, bins_df], axis=1)
