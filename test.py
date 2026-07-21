from sklearn.linear_model import LogisticRegression

def to_logit(probability):
    probability = np.clip(probability, 1e-6, 1 - 1e-6)

    return np.log(
        probability / (1 - probability)
    ).reshape(-1, 1)


def crossfit_calibration(raw_probability, target, groups):
    score = to_logit(raw_probability)
    calibrated = np.zeros(len(target))

    for train_idx, valid_idx in cv.split(
        score, target, groups
    ):
        calibrator = LogisticRegression()

        calibrator.fit(
            score[train_idx],
            target.iloc[train_idx]
        )

        calibrated[valid_idx] = calibrator.predict_proba(
            score[valid_idx]
        )[:, 1]

    # Фінальний calibrator для production-моделі
    final_calibrator = LogisticRegression()
    final_calibrator.fit(score, target)

    return calibrated, final_calibrator



q_ge_prem, calibrator_ge_prem = crossfit_calibration(
    raw_ge_prem,
    y_ge_prem,
    groups
)

q_hnwi, calibrator_hnwi = crossfit_calibration(
    raw_hnwi,
    y_hnwi,
    groups
)


# HNWI є підмножиною PREM+
q_hnwi = np.minimum(q_hnwi, q_ge_prem)

ordinal_probabilities = np.column_stack([
    1 - q_ge_prem,           # P(MASS)
    q_ge_prem - q_hnwi,      # P(PREM)
    q_hnwi                   # P(HNWI)
])

ordinal_probabilities = np.clip(
    ordinal_probabilities, 1e-9, 1
)

ordinal_probabilities /= ordinal_probabilities.sum(
    axis=1,
    keepdims=True
)

ordinal_metrics = evaluate_model(
    "Cumulative ordinal CatBoost",
    y,
    ordinal_probabilities
)