df_base = df.copy()

train_idx, val_idx = train_test_split(
    df_base.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=(df_base[TARGET_NAME] > 0)
)

df_train = df_base.loc[train_idx]
df_val = df_base.loc[val_idx]


y_train_clf = (df_train[TARGET_NAME] > 0).astype(int)
y_val_clf   = (df_val[TARGET_NAME] > 0).astype(int)

X_train_clf = df_train[features_to_use].copy()
X_val_clf   = df_val[features_to_use].copy()



df_train_reg = df_train[df_train[TARGET_NAME] > 0].copy()
df_val_reg   = df_val[df_val[TARGET_NAME] > 0].copy()

def preprocess_target(df_):
    df_ = df_.copy()

    df_ = df_[df_[TARGET_NAME] > 20]

    lower = df_[TARGET_NAME].quantile(0.1)
    upper = df_[TARGET_NAME].quantile(0.975)

    df_[TARGET_NAME] = df_[TARGET_NAME].clip(lower=lower)
    df_ = df_[df_[TARGET_NAME] < 6000]

    # якщо треба
    df_[TARGET_NAME] = np.log1p(df_[TARGET_NAME])

    return df_

df_train_reg = preprocess_target(df_train_reg)
df_val_reg   = preprocess_target(df_val_reg)


X_train_reg = df_train_reg[features_to_use].copy()
y_train_reg = df_train_reg[TARGET_NAME]

X_val_reg = df_val_reg[features_to_use].copy()
y_val_reg = df_val_reg[TARGET_NAME]