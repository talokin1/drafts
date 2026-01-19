import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def auto_select_features(X_train, y_train, X_val, y_val, cat_features, task='regression', threshold=0.0):
    """
    Автоматично відбирає фічі на основі Permutation Importance.
    
    Args:
        task: 'regression' (для R2) або 'classification' (для AUC)
        threshold: поріг важливості. Фічі з важливістю <= threshold будуть видалені.
                   Для жорсткої чистки став > 0 (наприклад, 0.001).
    """
    print(f"Starting feature selection. Initial count: {X_train.shape[1]}")
    
    # 1. Швидке навчання базової моделі
    params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    if task == 'regression':
        model = lgb.LGBMRegressor(objective='regression', metric='rmse', **params)
        scoring = 'r2'
    else:
        model = lgb.LGBMClassifier(objective='binary', metric='auc', class_weight='balanced', **params)
        scoring = 'roc_auc'

    # Важливо: fit робимо на train
    model.fit(
        X_train, y_train, 
        categorical_feature=cat_features,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    
    # 2. Permutation Importance (рахуємо на VAL сеті - це критично!)
    print(f"Calculating permutation importance on Validation set ({scoring})...")
    r = permutation_importance(
        model, X_val, y_val,
        n_repeats=5,       # Кількість прогонів для стабільності
        random_state=42,
        n_jobs=-1,
        scoring=scoring
    )
    
    # 3. Аналіз результатів
    importances = r.importances_mean
    feature_names = X_train.columns
    
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    # 4. Фільтрація
    # Залишаємо тільки ті, де важливість > порогу (і не від'ємна)
    selected_features = feature_imp_df[feature_imp_df['importance'] > threshold]['feature'].tolist()
    dropped_features = feature_imp_df[feature_imp_df['importance'] <= threshold]['feature'].tolist()
    
    print(f"\nDone. Kept {len(selected_features)} features. Dropped {len(dropped_features)}.")
    print(f"Top 5 features:\n{feature_imp_df.head(5)}")
    print(f"Worst 5 features (noise):\n{feature_imp_df.tail(5)}")
    
    return selected_features, feature_imp_df

# ==========================================
# ВИКОРИСТАННЯ (встав це перед основним навчанням)
# ==========================================

# 1. Для Регресора (найважливіше для R2)
# Використовуй ті ж маски, що я радив раніше (тільки багаті клієнти), або повний сет
good_features_reg, df_imp = auto_select_features(
    X_train_reg, y_train_reg,  # Твої підготовлені датасети для регресії
    X_val_reg, y_val_reg, 
    cat_features=cat_features,
    task='regression',
    threshold=0.0001 # Відсікаємо все, що не дає хоча б мінімального приросту R2
)

# 2. Оновлюємо датасети
X_train_reg_opt = X_train_reg[good_features_reg]
X_val_reg_opt = X_val_reg[good_features_reg]

# Тепер вчимо фінальну модель на X_train_reg_opt
print("Re-training final model with optimized features...")
# ... твій код reg.fit(X_train_reg_opt, ...)