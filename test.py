import optuna
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def objective(trial):
    # Простір гіперпараметрів
    param = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # Навчання з ранньою зупинкою
    # Важливо: використовуємо X_train, y_train_log (лог-таргет!)
    model = lgb.LGBMRegressor(**param)
    
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    # Передбачення
    preds_log = model.predict(X_val)
    mae = mean_absolute_error(y_val_log, preds_log)
    
    return mae

print("Starting Optuna optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30) # 30 ітерацій вистачить для початку

print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))