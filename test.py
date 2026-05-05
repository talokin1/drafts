import joblib

print("Збереження пайплайну...")

# Пакуємо обидві моделі та всі важливі константи в один словник
model_artifact = {
    "classifier": clf,
    "regressor": reg,
    "best_threshold": best_threshold, # Знайдений оптимальний поріг
    "global_cap": global_cap          # Знайдений ліміт для викидів
}

# Зберігаємо словник у файл
joblib.dump(model_artifact, "two_stage_income_model.pkl")

print("Модель успішно збережено у 'two_stage_income_model.pkl'")




import joblib
import numpy as np
import pandas as pd

def predict_potential_income(X_new, model_path="two_stage_income_model.pkl"):
    """
    Виконує передбачення потенційного доходу на основі збереженої Two-Stage моделі.
    
    X_new: pd.DataFrame з новими клієнтами. 
           ВАЖЛИВО: Назви колонок та їх типи мають повністю співпадати з X_train.
    """
    
    # 1. Завантаження артефакту
    artifact = joblib.load(model_path)
    clf = artifact["classifier"]
    reg = artifact["regressor"]
    best_threshold = artifact["best_threshold"]
    global_cap = artifact["global_cap"]
    
    # Перевірка, чи всі категоріальні фічі мають тип "category"
    # LightGBM вимагає цього для коректної роботи
    cat_cols = clf.booster_.pandas_categorical
    if cat_cols is not None:
        for c in cat_cols[0]:
            if c in X_new.columns:
                X_new[c] = X_new[c].astype("category")

    # ==========================================
    # 2. STAGE 1: Класифікація (Активний / Неактивний)
    # ==========================================
    # Отримуємо ймовірність класу 1 (активний)
    prob_active = clf.predict_proba(X_new)[:, 1]
    
    # Визначаємо, чи пройшов клієнт бар'єр
    is_active_pred = (prob_active >= best_threshold).astype(int)

    # ==========================================
    # 3. STAGE 2: Регресія (Оцінка прибутку)
    # ==========================================
    # Регресор робить прогноз для всіх (у логарифмічній шкалі)
    reg_preds_log = reg.predict(X_new)
    
    # Повертаємо з логарифма (експоненціюємо)
    reg_preds = np.expm1(reg_preds_log)
    
    # Обрізаємо від'ємні значення та гігантські викиди по кепу з трейну
    reg_preds_capped = np.clip(reg_preds, 0, global_cap)

    # ==========================================
    # 4. ОБ'ЄДНАННЯ (Згідно з твоєю логікою)
    # ==========================================
    # Якщо is_active_pred == 0, дохід множиться на 0.
    # Якщо is_active_pred == 1, беремо повний прогноз регресора.
    final_predictions = is_active_pred * reg_preds_capped
    
    return final_predictions

# Приклад використання:
# df_new = pd.read_csv("new_clients.csv")
# predictions = predict_potential_income(df_new)
# df_new["predicted_income"] = predictions