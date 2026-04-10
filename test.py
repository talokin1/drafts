import pandas as pd
import numpy as np
import joblib

# =========================
# 1. ПІДГОТОВКА BUCKETS
# =========================

# беремо тільки probability колонки (БЕЗ IDENTIFYCODE)
bucket_cols = external_liabs.loc[:, "0":">10M"].columns.to_list()

# привести до float
external_liabs[bucket_cols] = external_liabs[bucket_cols].apply(
    pd.to_numeric, errors="coerce"
)

# max bucket
external_liabs["max_bucket"] = external_liabs[bucket_cols].idxmax(axis=1)

# zero flag
external_liabs["is_zero_liabs"] = external_liabs["max_bucket"] == "0"


# =========================
# 2. MERGE В X
# =========================

X_ = X.merge(
    external_liabs[["IDENTIFYCODE", "is_zero_liabs"]],
    on="IDENTIFYCODE",
    how="left"
)

# fix NaN + тип
X_["is_zero_liabs"] = X_["is_zero_liabs"].fillna(False).astype(bool)


# =========================
# 3. SPLIT
# =========================

zero_clients = X_[X_["is_zero_liabs"]].copy()
non_zero_clients = X_[~X_["is_zero_liabs"]].copy()


# =========================
# 4. LOAD MODEL
# =========================

# ВАЖЛИВО: клас має бути оголошений
class LiabilitiesIncomeModel:
    def __init__(self, model, cat_cols, feature_cols):
        self.model = model
        self.cat_cols = cat_cols
        self.feature_cols = feature_cols

    def predict(self, X):
        X = X.copy()
        X = X[self.feature_cols]

        for c in self.cat_cols:
            X[c] = X[c].astype("category")

        y_log = self.model.predict(X)
        return np.expm1(y_log)


model = joblib.load(r"C:\Projects\DS-450 Corp_potential_income\scripts\models\pickle_models\Liabilities.pkl")


# =========================
# 5. PREDICT
# =========================

liabilities_pred = model.predict(non_zero_clients)

non_zero_clients["LIABILITIES_POTENTIAL"] = liabilities_pred
zero_clients["LIABILITIES_POTENTIAL"] = 0


# =========================
# 6. MERGE BACK
# =========================

final_liabs = pd.concat([zero_clients, non_zero_clients]).sort_index()

X["LIABILITIES_POTENTIAL"] = final_liabs["LIABILITIES_POTENTIAL"]