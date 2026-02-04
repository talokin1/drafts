import pandas as pd

# Отримуємо значення важливості
importance = reg.feature_importances_
feature_names = X_train_final.columns

# Створюємо DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values(by='importance', ascending=False)

# Зберігаємо у файл
feature_importance_df.to_csv('feature_importance.csv', index=False)
print("Feature importance збережено у feature_importance.csv")