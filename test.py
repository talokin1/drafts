# Отримуємо важливості з результату permutation_importance
importances = result.importances_mean
feature_names = X_test_reg.columns

# Створюємо DataFrame і сортуємо
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Дивимось на ТОП-10 і на "сміття"
print("--- ТОП-10 НАЙКРАЩИХ ФІЧ ---")
print(feature_importance_df.head(10))

print("\n--- КІЛЬКІСТЬ ШКІДЛИВИХ ФІЧ (Importance <= 0) ---")
print((feature_importance_df['importance'] <= 0).sum())


# Залишаємо тільки корисне
useful_features = feature_importance_df[feature_importance_df['importance'] > 0]['feature'].tolist()
print(f"Залишаємо {len(useful_features)} фіч із {len(feature_names)}")

# Перенавчаємо регресор ТІЛЬКИ на них
X_train_reg_opt = X_train_reg[useful_features]
X_test_opt = X_test[useful_features] # Для фінального тесту

regressor_opt = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42)
regressor_opt.fit(X_train_reg_opt, y_train_reg_log, categorical_feature=[c for c in useful_features if c in cat_features])