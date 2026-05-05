import joblib
import numpy as np
import pandas as pd

def predict_potential_income(X_new, model_path="two_stage_income_model.pkl"):
    # 1. Завантаження артефакту
    artifact = joblib.load(model_path)
    clf = artifact["classifier"]
    reg = artifact["regressor"]
    best_threshold = artifact["best_threshold"]
    global_cap = artifact["global_cap"]
    
    # ==========================================
    # КРОК 1.5: ФІЛЬТРАЦІЯ ТА ВИРІВНЮВАННЯ ФІЧ (ВИПРАВЛЕННЯ ПОМИЛКИ)
    # ==========================================
    # Дістаємо назви 165 колонок, на яких вчився класифікатор
    expected_features = clf.feature_name_
    
    # Перевіряємо, чи раптом не бракує якихось важливих колонок у нових даних
    missing_cols = set(expected_features) - set(X_new.columns)
    if missing_cols:
        raise ValueError(f"У нових даних бракує необхідних колонок: {missing_cols}")
        
    # Відсікаємо зайві колонки (залишаємо 165 з 503) і ставимо їх у правильному порядку!
    X_new = X_new[expected_features].copy()

    # Перевірка, чи всі категоріальні фічі мають тип "category"
    cat_cols = clf.booster_.pandas_categorical
    if cat_cols is not None:
        for c in cat_cols[0]:
            if c in X_new.columns:
                X_new[c] = X_new[c].astype("category")

    # ==========================================
    # 2. STAGE 1: Класифікація (Активний / Неактивний)
    # ==========================================
    prob_active = clf.predict_proba(X_new)[:, 1]
    is_active_pred = (prob_active >= best_threshold).astype(int)

    # ==========================================
    # 3. STAGE 2: Регресія (Оцінка прибутку)
    # ==========================================
    reg_preds_log = reg.predict(X_new)
    reg_preds = np.expm1(reg_preds_log)
    reg_preds_capped = np.clip(reg_preds, 0, global_cap)

    # ==========================================
    # 4. ОБ'ЄДНАННЯ
    # ==========================================
    final_predictions = is_active_pred * reg_preds_capped
    
    return final_predictions