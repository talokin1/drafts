ratio = len(y_train[y_train == 0]) / len(y_train[y_train > 0]) # Автоматичний розрахунок ~60

weights = np.where(y_train_log > 0, ratio, 1)

reg = lgb.LGBMRegressor(
    objective="regression", # Див. пораду нижче про Tweedie
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=5, # Можна зменшити, бо у вас мало "цінних" даних
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42, # Краще зафіксувати числом замість змінної для відтворюваності тут
    n_jobs=-1
)

reg.fit(
    X_train_final,
    y_train_log,
    sample_weight=weights,
    eval_set=[(X_val_final, y_val_log)],
    eval_metric="l1", # Або 'mae'
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
)

y_pred_log = reg.predict(X_val_final)