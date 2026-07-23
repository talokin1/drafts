feature_cols = X.columns.tolist()

final_models = []

for seed in SEEDS:
    model = create_model(seed)
    model.fit(X[feature_cols], y, cat_features=cat_cols)
    final_models.append(model)


def predict_hnwi(df, models=final_models, threshold=hnwi_threshold):
    data = df.copy()

    for col in cat_cols: data[col] = data[col].astype("string").fillna("Missing").astype(str)

    probability = np.mean([model.predict_proba(data[feature_cols])[:, 1] for model in models], axis=0)
    return pd.DataFrame({"MOBILEPHONE": data["MOBILEPHONE"].values, "is_hnwi": (probability >= threshold).astype(int)})


hnwi_predictions = predict_hnwi(client_df_inf)
hnwi_predictions.head()







def predict_hnwi(df, models=final_models, threshold=hnwi_threshold):
    data = df.copy()

    for col in cat_cols: data[col] = data[col].astype("string").fillna("Missing").astype(str)

    probability = np.mean([model.predict_proba(data[feature_cols])[:, 1] for model in models], axis=0)
    return pd.DataFrame({"MOBILEPHONE": data["MOBILEPHONE"].values, "hnwi_probability": probability, "is_hnwi": (probability >= threshold).astype(int)})