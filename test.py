import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, median_absolute_error

# 1. Повертаємо цільову змінну в оригінальний масштаб ДЛЯ ТРЕНУВАННЯ
# (Оскільки y_train_log у тебе вже був прологарифмований раніше)
y_train_orig = np.expm1(y_train_log)
y_val_orig = np.expm1(y_val_log)

print("--- Training Single Model with Huber Loss ---")
reg_huber = lgb.LGBMRegressor(
    objective="huber",
    alpha=1.5, # Поріг переходу від MSE до MAE. Чим менше, тим стійкіша до викидів.
    n_estimators=3000,
    learning_rate=0.015, # Робимо крок меншим для стабільності на оригінальному масштабі
    num_leaves=31,       # Можна дати трохи більше свободи, бо це єдина модель
    min_child_samples=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Навчаємо модель безпосередньо на оригінальних грошах
reg_huber.fit(
    X_train_final, y_train_orig,
    eval_set=[(X_val_final, y_val_orig)],
    eval_metric="mae", # Валідуємося все одно по MAE
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=True)]
)

# 2. Оцінка
pred_huber = reg_huber.predict(X_val_final)

# Захист від від'ємних прогнозів (регресія може видати невеликий мінус)
pred_huber = np.maximum(pred_huber, 0)

final_mae = mean_absolute_error(y_val_orig, pred_huber)
final_medae = median_absolute_error(y_val_orig, pred_huber)

print("-" * 40)
print(f"FINAL ORIGINAL MAE (Huber)  : {final_mae:,.2f}")
print(f"FINAL ORIGINAL MedAE (Huber): {final_medae:,.2f}")

# Дивимося на результати в DataFrame
validation_results_huber = pd.DataFrame({
    'IDENTIFYCODE': X_val.index,
    'True_Value': y_val_orig,
    'Predicted': pred_huber
})