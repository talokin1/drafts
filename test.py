import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "fx_hurdle_model.pkl"

num_cols = [c for c in final_features if c not in cat_cols]

num_medians = {}

for c in num_cols:
    if c in df_train.columns:
        num_medians[c] = pd.to_numeric(df_train[c], errors="coerce").median()
    else:
        num_medians[c] = 0

category_values = {}

for c in cat_cols:
    if c in X_train_clf.columns:
        if hasattr(X_train_clf[c], "cat"):
            cats = list(X_train_clf[c].cat.categories.astype(str))
        else:
            cats = list(pd.Series(X_train_clf[c].astype("string")).dropna().unique())

        if "UNKNOWN" not in cats:
            cats.append("UNKNOWN")

        category_values[c] = cats

model_pack = {
    "clf_binary": clf_binary,
    "reg": reg,

    "final_features": final_features,
    "cat_cols": cat_cols,
    "num_cols": num_cols,

    "num_medians": num_medians,
    "category_values": category_values,

    "target_name": TARGET_NAME,
    "id_col": ID_COL,
    "fx_upper_cap": FX_UPPER_CAP,
    "random_state": RANDOM_STATE
}

joblib.dump(model_pack, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")























import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "fx_hurdle_model.pkl"

model_pack = joblib.load(MODEL_PATH)

clf_binary = model_pack["clf_binary"]
reg = model_pack["reg"]

final_features = model_pack["final_features"]
cat_cols = model_pack["cat_cols"]
num_cols = model_pack["num_cols"]

num_medians = model_pack["num_medians"]
category_values = model_pack["category_values"]

TARGET_NAME = model_pack["target_name"]
ID_COL = model_pack["id_col"]
FX_UPPER_CAP = model_pack["fx_upper_cap"]

print("Model loaded")
print("Number of features:", len(final_features))
print("Categorical columns:", cat_cols)