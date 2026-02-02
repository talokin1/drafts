# 1. Відбираємо тільки корисні фічі (де важливість > 0)
# Або ще жорсткіше: беремо Тільки Топ-70, щоб прибрати "хвіст"
selected_features = feature_imp.sort_values(by="Value", ascending=False).head(70)['Feature'].tolist()

print(f"Keeping top {len(selected_features)} features.")

# 2. Створюємо нові тренувальні набори
X_train_sel = X_train[selected_features]
X_val_sel = X_val[selected_features]

# 3. Перетреновуємо модель (вже без L1 регуляризації, бо ми самі відібрали фічі)
final_reg = lgb.LGBMRegressor(
    objective="huber",       # Залишаємо Huber, він добре себе показав
    metric="mae",
    alpha=1.5,
    
    n_estimators=3000,       # Можна трохи менше, бо фічей менше
    learning_rate=0.05,      # Трохи агресивніше навчання
    num_leaves=31,
    
    # Зменшуємо регуляризацію, бо сміття вже немає
    reg_alpha=1,             
    reg_lambda=1,
    
    random_state=42,
    n_jobs=-1
)

final_reg.fit(
    X_train_sel, 
    y_train_log,
    categorical_feature=[c for c in cat_cols if c in selected_features], # Тільки якщо категорія потрапила в топ
    eval_set=[(X_val_sel, y_val_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
)

# 4. Дивимось фінальний результат
y_pred_final_log = final_reg.predict(X_val_sel)
print(f"Final R2: {r2_score(y_val_log, y_pred_final_log):.4f}")
print(f"Final MAE (log): {mean_absolute_error(y_val_log, y_pred_final_log):.4f}")