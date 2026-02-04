import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

RANDOM_STATE = 42

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Базові категорії
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

# KVED-derived
X_train["KVED_DIV"] = X_train["KVED"].apply(lambda x: x.split(".")[0] if "." in x else x[:2])
X_train["KVED_GROUP"] = X_train["KVED"].apply(lambda x: x[:4] if "." in x else x[:3])

X_val["KVED_DIV"] = X_val["KVED"].apply(lambda x: x.split(".")[0] if "." in x else x[:2])
X_val["KVED_GROUP"] = X_val["KVED"].apply(lambda x: x[:4] if "." in x else x[:3])

for c in ["KVED_DIV", "KVED_GROUP"]:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")


te_cols = ["KVED_DIV", "KVED_GROUP", "OPF_CODE", "SECTION_CODE", "FIRM_TYPE"]

TE_CONFIG = {
    "KVED_DIV": 50,
    "KVED_GROUP": 30,
    "OPF_CODE": 10,
    "SECTION_CODE": 20,
    "FIRM_TYPE": 5
}


def kfold_target_encoding(X, y, cols, te_config, n_splits=5, random_state=42):
    X_out = X.copy()
    global_mean = y.mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in cols:
        m = te_config[col]
        te_name = f"{col}_TE"
        X_out[te_name] = np.nan

        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr = y.iloc[tr_idx]

            stats = (
                pd.DataFrame({col: X_tr[col], "target": y_tr})
                .groupby(col)["target"]
                .agg(["count", "mean"])
            )

            smooth = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)

            X_out.iloc[val_idx, X_out.columns.get_loc(te_name)] = (
                X_val[col].map(smooth)
            )

        X_out[te_name].fillna(global_mean, inplace=True)

    return X_out

X_train_te = kfold_target_encoding(
    X_train,
    y_train,
    cols=te_cols,
    te_config=TE_CONFIG,
    n_splits=5
)

# Validation — БЕЗ leakage
def apply_te_val(X_train, y_train, X_val, cols, te_config):
    X_val_out = X_val.copy()
    global_mean = y_train.mean()

    for col in cols:
        m = te_config[col]
        stats = (
            pd.DataFrame({col: X_train[col], "target": y_train})
            .groupby(col)["target"]
            .agg(["count", "mean"])
        )
        smooth = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
        X_val_out[f"{col}_TE"] = X_val[col].map(smooth).fillna(global_mean)

    return X_val_out

X_val_te = apply_te_val(X_train, y_train, X_val, te_cols, TE_CONFIG)

X_train_final = X_train_te.drop(columns=te_cols)
X_val_final = X_val_te.drop(columns=te_cols)

reg = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

reg.fit(
    X_train_final,
    y_train,
    eval_set=[(X_val_final, y_val)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(200)]
)
