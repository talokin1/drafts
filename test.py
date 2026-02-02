import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 0. ЕТАП: Функції перевірки однорідності (Твій код)
# ==========================================
def check_distribution(y_train, y_val, threshold=0.05):
    """
    Перевіряє, чи однакові розподіли у тренувальній та валідаційній вибірках.
    Повертає True, якщо p-value обох тестів вищі за поріг (розподіли не відрізняються).
    """
    # Тест Колмогорова-Смірнова (форми розподілів)
    ks_stat, ks_p = ks_2samp(y_train, y_val)
    
    # Тест Манна-Вітні (медіани/ранги)
    mw_stat, mw_p = mannwhitneyu(y_train, y_val, alternative='two-sided')
    
    return ks_p >= threshold and mw_p >= threshold, ks_p, mw_p

def robust_split_search(X, y, test_size=0.2, threshold=0.05, max_attempts=100):
    """
    Шукає random_state, який дає однорідне розбиття.
    """
    print(f"--- Searching for homogeneous split (Threshold p-val > {threshold}) ---")
    
    best_seed = None
    best_p_min = -1
    best_split = None
    
    for seed in range(42, 42 + max_attempts):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        is_valid, ks_p, mw_p = check_distribution(y_tr, y_val, threshold=threshold)
        min_p = min(ks_p, mw_p)
        
        # Якщо знайшли ідеальний спліт - одразу повертаємо
        if is_valid:
            print(f"✅ Found VALID split! Seed: {seed}")
            print(f"   KS p-value: {ks_p:.4f}, MW p-value: {mw_p:.4f}")
            return X_tr, X_val, y_tr, y_val, seed
        
        # Зберігаємо "найкращий з гірших" на випадок невдачі
        if min_p > best_p_min:
            best_p_min = min_p
            best_seed = seed
            best_split = (X_tr, X_val, y_tr, y_val)

    print(f"⚠️ Warning: Perfect split not found after {max_attempts} attempts.")
    print(f"   Using best found (Seed {best_seed}) with min p-value: {best_p_min:.4f}")
    return (*best_split, best_seed)

# ==========================================
# 1. ЕТАП: Підготовка даних
# ==========================================
# Припускаємо, що df вже завантажений і очищений
target_col = "CURR_ACC"

# Видаляємо таргет з X (і категорії переводимо в category тип)
X = df.drop(columns=[target_col], errors='ignore').copy()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_features:
    X[c] = X[c].astype('category')

# Готуємо y (логарифм)
y = np.log1p(df[target_col].clip(lower=0))

# --- ЗАПУСК РОЗУМНОГО СПЛІТА ---
# Використовуємо твою логіку для пошуку ідеального random_state
X_train, X_val, y_train_log, y_val_log, used_seed = robust_split_search(
    X, y, test_size=0.2, threshold=0.05  # Можна поставити 0.1 або 0.2 для суворості
)

# ==========================================
# 2. ЕТАП: Selector Model (Відбір фічей)
# ==========================================
print("\n--- Training Selector Model ---")
selector_reg = lgb.LGBMRegressor(
    objective="huber",
    metric="mae",
    alpha=1.5,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    reg_alpha=10, 
    reg_lambda=10,
    random_state=used_seed, # Використовуємо знайдений сід!
    n_jobs=-1,
    verbose=-1
)

selector_reg.fit(
    X_train, y_train_log,
    categorical_feature=cat_features,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

# Відбираємо Топ-50 фічей
feature_imp = pd.DataFrame({
    'Value': selector_reg.feature_importances_,
    'Feature': X_train.columns
}).sort_values(by="Value", ascending=False)

selected_features = feature_imp.head(50)['Feature'].tolist() # Топ-50 достатньо
print(f"Selected {len(selected_features)} features.")

# ==========================================
# 3. ЕТАП: Фінальна модель
# ==========================================
print("\n--- Training Final Model ---")
X_train_sel = X_train[selected_features].copy()
X_val_sel = X_val[selected_features].copy()
cat_features_sel = [c for c in cat_features if c in selected_features]

final_reg = lgb.LGBMRegressor(
    objective="huber",
    metric="mae",
    alpha=1.5,
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    reg_alpha=1, # Слабша регуляризація
    reg_lambda=1,
    random_state=used_seed,
    n_jobs=-1,
    verbose=-1
)

final_reg.fit(
    X_train_sel, y_train_log,
    categorical_feature=cat_features_sel,
    eval_set=[(X_val_sel, y_val_log)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
)

# Оцінка
y_pred_log = final_reg.predict(X_val_sel)
print(f"FINAL R2: {r2_score(y_val_log, y_pred_log):.4f}")
print(f"FINAL MAE: {mean_absolute_error(y_val_log, y_pred_log):.4f}")