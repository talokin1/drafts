mask = y_test_cls == 1

mae_vip = mean_absolute_error(
    np.expm1(y_test_log[mask]),
    final_preds[mask]
)

r2_vip = r2_score(
    np.expm1(y_test_log[mask]),
    final_preds[mask]
)
