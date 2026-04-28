df_pred = predict_bucket_expected_income(
    df_new=df.copy(),
    model_path=MODEL_PATH
)

# Перевірка, що кількість рядків та індекси збігаються
print("df shape:", df.shape)
print("df_pred shape:", df_pred.shape)

print("same index:", df.index.equals(df_pred.index))

# Безпечне присвоєння
df["LIABILITIES_POTENTIAL"] = df_pred["LIABILITIES_POTENTIAL"]