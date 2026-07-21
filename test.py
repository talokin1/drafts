def create_binary_model(seed):
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=500,
        depth=4,
        learning_rate=0.04,
        l2_leaf_reg=10,
        random_strength=1,
        auto_class_weights="SqrtBalanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False
    )


def binary_oof(X, target, groups, seed):
    probabilities = np.zeros(len(X))

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, target, groups)
    ):
        model = create_binary_model(seed + fold)

        model.fit(
            X.iloc[train_idx],
            target.iloc[train_idx],
            cat_features=cat_cols
        )

        probabilities[valid_idx] = model.predict_proba(
            X.iloc[valid_idx]
        )[:, 1]

    return probabilities


y_ge_prem = (y >= 1).astype(int)  # PREM або HNWI
y_hnwi = (y == 2).astype(int)     # тільки HNWI

raw_ge_prem = binary_oof(
    X, y_ge_prem, groups, seed=100
)

raw_hnwi = binary_oof(
    X, y_hnwi, groups, seed=200
)