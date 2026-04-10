probs_multi_full = clf_model.predict_proba(X_val_clf)

y_expected_full = np.zeros(len(X_val_clf))

for i in range(probs_multi_full.shape[1]):
    if i in bucket_medians.index:
        y_expected_full += probs_multi_full[:, i] * bucket_medians.loc[i]

y_expected_full = np.expm1(y_expected_full)

p_income = probs_clf  # вже (5308,)
y_expected = y_expected_full  # тепер теж (5308,)

final_prediction = p_income * y_expected


y_val_true_full = df.loc[X_val_clf.index, TARGET_NAME]

mae_final = mean_absolute_error(y_val_true_full, final_prediction)

print("\n=== FINAL MODEL ===")
print("MAE (P * E[y]):", mae_final)
