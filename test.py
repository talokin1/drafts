false_positive_idx = np.where((oof_prediction == 1) & (y_hnwi == 0))[0]

false_positives = (
    X.iloc[false_positive_idx]
    .assign(
        true_class=y.iloc[false_positive_idx].map({0: "MASS", 1: "PREM", 2: "HNWI"}),
        p_affluent=oof_p_affluent[false_positive_idx],
        p_hnwi_given_affluent=oof_p_hnwi_conditional[false_positive_idx],
        hnwi_score=oof_hnwi_score[false_positive_idx],
    )
    .sort_values("hnwi_score", ascending=False)
)

display(false_positives)