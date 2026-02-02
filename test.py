import seaborn as sns

# 1. Рахуємо матрицю кореляцій (тільки для відібраних фічей)
# Використовуємо X_train_sel, який ми створили раніше
corr_matrix = X_train_sel.corr().abs()

# 2. Малюємо хітмап (щоб оцінити масштаб проблеми очима)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Correlation Matrix of Top Features")
plt.show()

# 3. Знаходимо пари з кореляцією > 0.95
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(f"Features with correlation > 0.95: {len(to_drop)}")
print("Example dropped features:", to_drop[:5])

# 4. Видаляємо їх
X_train_final = X_train_sel.drop(columns=to_drop)
X_val_final = X_val_sel.drop(columns=to_drop)

print(f"Final feature count: {X_train_final.shape[1]}")