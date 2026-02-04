from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def kfold_target_encoding(X, y, cols, te_config, n_splits=5, random_state=42):
    X_out = X.copy()
    global_mean = float(y.mean())

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in cols:
        m = te_config[col]
        te_name = f"{col}_TE"

        # важливо: явний float dtype
        X_out[te_name] = np.nan
        X_out[te_name] = X_out[te_name].astype(float)

        for tr_idx, val_idx in kf.split(X_out):
            tr_index = X_out.index[tr_idx]
            val_index = X_out.index[val_idx]

            X_tr = X_out.loc[tr_index, col]
            y_tr = y.loc[tr_index]

            stats = (
                pd.DataFrame({col: X_tr, "target": y_tr})
                .groupby(col)["target"]
                .agg(["count", "mean"])
            )

            smooth = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)

            X_out.loc[val_index, te_name] = X_out.loc[val_index, col].map(smooth).astype(float)

        X_out[te_name] = X_out[te_name].fillna(global_mean)

    return X_out


def apply_te_val(X_train, y_train, X_val, cols, te_config):
    X_val_out = X_val.copy()
    global_mean = float(y_train.mean())

    for col in cols:
        m = te_config[col]
        te_name = f"{col}_TE"

        stats = (
            pd.DataFrame({col: X_train[col], "target": y_train})
            .groupby(col)["target"]
            .agg(["count", "mean"])
        )

        smooth = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)

        X_val_out[te_name] = X_val_out[col].map(smooth).astype(float).fillna(global_mean)

    return X_val_out





X_train_te = kfold_target_encoding(X_train, y_train_log, te_cols, TE_CONFIG, n_splits=5)
X_val_te = apply_te_val(X_train, y_train_log, X_val, te_cols, TE_CONFIG)

X_train_final = X_train_te.drop(columns=te_cols)
X_val_final = X_val_te.drop(columns=te_cols)

reg.fit(
    X_train_final,
    y_train_log,
    eval_set=[(X_val_final, y_val_log)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(200)]
)

y_pred_log = reg.predict(X_val_final)
