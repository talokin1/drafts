df_base = df.copy()

y_clf_full = (df_base[TARGET_NAME] > 0).astype(int)

train_idx, val_idx = train_test_split(
    df_base.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_clf_full
)

df_train = df_base.loc[train_idx].copy()
df_val   = df_base.loc[val_idx].copy()

def build_features(df, fitted_params=None):
    df = df.copy()

    # --- numeric cols
    all_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if TARGET_NAME in all_numeric_cols:
        all_numeric_cols.remove(TARGET_NAME)

    categorized_cols = set(scale_cols + diff_cols + ratio_cols + count_cols)
    raw_cols = [c for c in all_numeric_cols if (c not in categorized_cols) and (c not in liabs_cols)]

    num_cols_all = scale_cols + diff_cols + ratio_cols + count_cols + raw_cols + liabs_cols

    # --- zero rate
    if fitted_params is None:
        zero_rate = (df[num_cols_all] == 0).mean()
        valid_feats = set(zero_rate[zero_rate < 0.95].index)
    else:
        valid_feats = fitted_params["valid_feats"]

    # --- відбір
    real_cols = [c for c in num_cols_all if c in valid_feats]

    # --- skew transform
    processed_cols = []

    if fitted_params is None:
        skew_map = {}
        for col in real_cols:
            val_skew = df[col].dropna().skew()
            skew_map[col] = abs(val_skew) > SKEW_THRESHOLD
    else:
        skew_map = fitted_params["skew_map"]

    for col in real_cols:
        if skew_map[col]:
            new_col = col + "_log"
            df[new_col] = signed_log1p(df[col])
            processed_cols.append(new_col)
        else:
            processed_cols.append(col)

    # --- фінальні фічі
    features = processed_cols + cat_cols

    if fitted_params is None:
        return df, {
            "valid_feats": valid_feats,
            "skew_map": skew_map,
            "features": features
        }
    else:
        return df, fitted_params
    
df_train_proc, fitted_params = build_features(df_train)
df_val_proc, _ = build_features(df_val, fitted_params)

X_train_full = df_train_proc[fitted_params["features"]]

X_train_wo_corr, removed_features = remove_highly_correlated_features(
    X_train_full.select_dtypes(include=['number'])
)

final_features = list(X_train_wo_corr.columns) + [
    c for c in fitted_params["features"] if c in cat_cols
]

# застосовуємо до val
X_val_full = df_val_proc[final_features]



X_train_clf = df_train_proc[final_features]
X_val_clf   = df_val_proc[final_features]

y_train_clf = (df_train[TARGET_NAME] > 0).astype(int)
y_val_clf   = (df_val[TARGET_NAME] > 0).astype(int)

df_train_reg = df_train_proc[df_train[TARGET_NAME] > 0].copy()
df_val_reg   = df_val_proc[df_val[TARGET_NAME] > 0].copy()





df_train_reg = preprocess_target(df_train_reg)
df_val_reg   = preprocess_target(df_val_reg)

X_train_reg = df_train_reg[final_features]
X_val_reg   = df_val_reg[final_features]

y_train_reg = df_train_reg[TARGET_NAME]
y_val_reg   = df_val_reg[TARGET_NAME]