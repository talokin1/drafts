import joblib

artifact = {
    "model": reg,
    "features": features_to_use,
    "cat_cols": cat_cols,
}

joblib.dump(artifact, "income_model.pkl")


import numpy as np
import pandas as pd
import joblib


# =========================
# LOAD MODEL
# =========================
artifact = joblib.load("income_model.pkl")

model = artifact["model"]
features = artifact["features"]
cat_cols = artifact["cat_cols"]


# =========================
# TRANSFORMS (ТІ Ж САМІ!)
# =========================
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def apply_transforms(df):
    df = df.copy()

    for col in df.columns:
        if col.endswith(("_CUR", "_PREV", "_DIF")):
            if col in df:
                df[col] = signed_log1p(df[col])

        elif col == "NB_EMPL":
            df[col] = np.log1p(df[col])

    return df


# =========================
# PREPARE DATA
# =========================
def prepare_X(df):
    df = df.copy()

    # трансформації
    df = apply_transforms(df)

    # залишаємо тільки потрібні фічі
    X = df[features].copy()

    # категоріальні
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return X


# =========================
# PREDICT
# =========================
def predict(df):
    X = prepare_X(df)

    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)

    return y_pred


# =========================
# USAGE
# =========================
# df_new = pd.read_csv("new_clients.csv")

# preds = predict(df_new)

# df_new["ACCOUNTS_POTENTIAL"] = preds

# df_new[["IDENTIFYCODE", "ACCOUNTS_POTENTIAL"]].to_csv("predictions.csv", index=False)