import pymc as pm

from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


num_cols = [
    col for col in X.columns
    if col not in cat_cols
]


def bayesian_ordered_oof(X, y, groups):
    probabilities = np.zeros((len(X), 3))

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y, groups)
    ):
        preprocessor = ColumnTransformer([
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler()
                ),
                num_cols
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=5,
                        sparse_output=False
                    )
                ),
                cat_cols
            )
        ])

        X_train = preprocessor.fit_transform(
            X.iloc[train_idx]
        ).astype(float)

        X_valid = preprocessor.transform(
            X.iloc[valid_idx]
        ).astype(float)

        y_train = y.iloc[train_idx].to_numpy()

        with pm.Model() as model:
            beta = pm.Normal(
                "beta",
                mu=0,
                sigma=0.5,
                shape=X_train.shape[1]
            )

            cutpoints = pm.Normal(
                "cutpoints",
                mu=0,
                sigma=1.5,
                shape=2,
                transform=pm.distributions.transforms.ordered,
                initval=np.array([-0.5, 0.5])
            )

            eta = pm.math.dot(X_train, beta)

            pm.OrderedLogistic(
                "segment",
                eta=eta,
                cutpoints=cutpoints,
                observed=y_train
            )

            trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                target_accept=0.9,
                random_seed=42 + fold,
                progressbar=False
            )

        beta_samples = (
            trace.posterior["beta"]
            .values
            .reshape(-1, X_train.shape[1])
        )

        cutpoint_samples = (
            trace.posterior["cutpoints"]
            .values
            .reshape(-1, 2)
        )

        eta_valid = X_valid @ beta_samples.T

        p_mass = expit(
            cutpoint_samples[:, 0][None, :] - eta_valid
        ).mean(axis=1)

        p_hnwi = expit(
            eta_valid - cutpoint_samples[:, 1][None, :]
        ).mean(axis=1)

        p_prem = 1 - p_mass - p_hnwi

        fold_probabilities = np.column_stack([
            p_mass,
            p_prem,
            p_hnwi
        ])

        fold_probabilities = np.clip(
            fold_probabilities, 1e-9, None
        )

        fold_probabilities /= fold_probabilities.sum(
            axis=1,
            keepdims=True
        )

        probabilities[valid_idx] = fold_probabilities

    return probabilities

bayesian_probabilities = bayesian_ordered_oof(
    X, y, groups
)

bayesian_metrics = evaluate_model(
    "Bayesian Ordered Logistic",
    y,
    bayesian_probabilities
)








comparison = pd.DataFrame({
    "Multiclass CatBoost": multiclass_metrics,
    "Ordinal CatBoost": ordinal_metrics,
    "Bayesian Ordered Logistic": bayesian_metrics
}).T

display(
    comparison.sort_values(
        "HNWI_average_precision",
        ascending=False
    )
)



final_ge_prem_model = create_binary_model(seed=501)
final_hnwi_model = create_binary_model(seed=502)

final_ge_prem_model.fit(
    X,
    y_ge_prem,
    cat_features=cat_cols
)

final_hnwi_model.fit(
    X,
    y_hnwi,
    cat_features=cat_cols
)