import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Очистка від "технічних" мінусів (овердрафтів)
df['CURR_ACC'] = df['CURR_ACC'].clip(lower=0)

# Сітка для пошуку ідеального варіанту
# Додав 10 (мій фаворит) і 50 (більш консервативний)
thresholds = [5, 10, 25, 50] 

best_mae = float('inf')
best_threshold = 0
results = {}

print("Починаємо пошук оптимального порогу...")

for t in thresholds:
    # КРОК A: Класифікація "Чи є значуща сума?"
    # 1 - це клієнт з грошима (> t), 0 - пустий або "сміттєвий" залишок
    y_class_temp = (df['CURR_ACC'] > t).astype(int)
    
    # Стратифікований спліт
    X_train, X_test, y_cls_train, y_cls_test = train_test_split(
        df.drop(columns=['CURR_ACC']), y_class_temp,
        test_size=0.2, random_state=42, stratify=y_class_temp
    )
    
    # КРОК B: Навчання Класифікатора
    clf = lgb.LGBMClassifier(n_estimators=200, random_state=42, class_weight='balanced', verbose=-1)
    clf.fit(X_train, y_cls_train, categorical_feature=cat_features)
    
    # КРОК C: Навчання Регресора (ТІЛЬКИ на "живих" клієнтах з трейну)
    mask_vip_train = y_cls_train == 1
    X_reg_train = X_train[mask_vip_train]
    
    # Важливо: ми вчимо регресор передбачати точну суму
    # Але оскільки ми відсікли < t, дані будуть чистішими
    y_reg_train_log = np.log1p(df.loc[X_reg_train.index, 'CURR_ACC'])
    
    reg = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    reg.fit(X_reg_train, y_reg_train_log, categorical_feature=cat_features)
    
    # КРОК D: Валідація (Two-Stage Prediction)
    # 1. Ймовірність
    prob_active = clf.predict_proba(X_test)[:, 1]
    
    # 2. Прогноз суми (для всіх, потім занулимо)
    pred_log = reg.predict(X_test)
    pred_amount = np.expm1(pred_log)
    
    # 3. Комбінація (Soft Gating)
    # Формула: Ймовірність * Прогноз суми
    # Це "очікувана вартість" (Expected Value)
    final_pred = prob_active * pred_amount
    
    # Рахуємо реальний MAE
    y_true = df.loc[X_test.index, 'CURR_ACC']
    mae = mean_absolute_error(y_true, final_pred)
    
    results[t] = mae
    print(f"Threshold {t} грн -> MAE: {mae:.2f}")
    
    if mae < best_mae:
        best_mae = mae
        best_threshold = t

print(f"\n🏆 Переможець: {best_threshold} грн (MAE: {best_mae:.2f})")