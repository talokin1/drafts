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