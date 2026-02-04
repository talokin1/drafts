from sklearn.model_selection import KFold

def kfold_target_encoding(
    X,
    y,
    cols,
    te_config,
    n_splits=5,
    random_state=42
):
    X_te = X.copy()
    global_mean = y.mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in cols:
        m = te_config[col]
        te_col = f"{col}_TE"
        X_te[te_col] = np.nan

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]

            stats = (
                pd.DataFrame({col: X_tr[col], "target": y_tr})
                .groupby(col)["target"]
                .agg(["count", "mean"])
            )

            smooth = (
                (stats["count"] * stats["mean"] + m * global_mean)
                / (stats["count"] + m)
            )

            X_te.iloc[val_idx, X_te.columns.get_loc(te_col)] = (
                X_val[col].map(smooth)
            )

        X_te[te_col].fillna(global_mean, inplace=True)

    return X_te

def apply_te_on_validation(X_train, y_train, X_val, cols, te_config):
    X_val_te = X_val.copy()
    global_mean = y_train.mean()

    for col in cols:
        m = te_config[col]
        stats = (
            pd.DataFrame({col: X_train[col], "target": y_train})
            .groupby(col)["target"]
            .agg(["count", "mean"])
        )

        smooth = (
            (stats["count"] * stats["mean"] + m * global_mean)
            / (stats["count"] + m)
        )

        X_val_te[f"{col}_TE"] = X_val[col].map(smooth).fillna(global_mean)

    return X_val_te









te_cols = ["KVED_DIV", "KVED_GROUP", "OPF_CODE", "SECTION_CODE", "FIRM_TYPE"]

X_train_te = kfold_target_encoding(
    X_train,
    y_train_log,
    cols=te_cols,
    te_config=TE_CONFIG,
    n_splits=5
)

X_train_final = X_train_te.drop(columns=te_cols)
X_val_final   = X_val_te.drop(columns=te_cols)

cat_features = [
    c for c in X_train_final.columns
    if X_train_final[c].dtype.name == "category"
]
