import optuna
from sklearn.metrics import mean_absolute_error

def objective(trial):

    clf_params = {
        "objective": "binary",
        "learning_rate": trial.suggest_float("clf_lr", 0.01, 0.07),
        "num_leaves": trial.suggest_int("clf_leaves", 16, 64),
        "min_child_samples": trial.suggest_int("clf_min_child", 20, 300),
        "subsample": trial.suggest_float("clf_subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("clf_colsample", 0.6, 0.95),
        "reg_lambda": trial.suggest_float("clf_lambda", 0, 10),
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1
    }

    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(
        X_train, y_cls_train,
        categorical_feature=cat_features,
        eval_set=[(X_val, y_cls_val)],
        callbacks=[lgb.early_stopping(50)],
        verbose=False
    )

    # ---------- Stage 2 ----------
    reg_params = {
        "objective": "regression_l1",
        "learning_rate": trial.suggest_float("reg_lr", 0.01, 0.07),
        "num_leaves": trial.suggest_int("reg_leaves", 32, 256),
        "min_child_samples": trial.suggest_int("reg_min_child", 50, 500),
        "subsample": trial.suggest_float("reg_subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("reg_colsample", 0.6, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "n_estimators": 3000,
        "random_state": 42,
        "n_jobs": -1
    }

    reg = lgb.LGBMRegressor(**reg_params)
    reg.fit(
        X_train_reg, y_train_reg,
        categorical_feature=cat_features,
        eval_set=[(X_val_reg, y_val_reg)],
        callbacks=[lgb.early_stopping(50)],
        verbose=False
    )

    # ---------- Threshold ----------
    threshold = trial.suggest_float("threshold", 0.2, 0.7)

    # ---------- Prediction ----------
    probs = clf.predict_proba(X_val)[:, 1]
    preds = np.zeros(len(X_val))

    mask = probs > threshold
    if mask.sum() > 0:
        preds_log = reg.predict(X_val[mask])
        preds[mask] = np.expm1(preds_log)

    mae = mean_absolute_error(y_val, preds)
    return mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
