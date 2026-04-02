import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score

# 1. Підготовка даних
# Видаляємо ідентифікатор, він не несе предиктивної сили
X = train_dataset.drop(columns=['MOBILEPHONE', 'is_hnwi'])

# Перетворюємо таргет у формат 1/0
y = train_dataset['is_hnwi'].astype(int)

# Знаходимо категоріальні та булеві колонки
# CatBoost любить, коли булеві або текстові дані явно вказані
cat_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Заповнюємо пропуски в категоріях (CatBoost сам впорається з NaN у числових)
X[cat_features] = X[cat_features].fillna('Missing').astype(str)

# 2. Налаштування крос-валідації
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Списки для збереження метрик по кожному фолду
metrics = {'pr_auc': [], 'precision': [], 'recall': [], 'f2_score': []}

print("Починаємо навчання з крос-валідацією...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 3. Ініціалізація моделі
    # Використовуємо 'Balanced' для дисбалансу та жорсткі обмеження від перенавчання
    model = CatBoostClassifier(
        iterations=300,             # Невелика кількість дерев
        depth=3,                    # Дуже "мілкі" дерева (всього 8 листків)
        learning_rate=0.03,         # Повільне навчання
        l2_leaf_reg=20,             # Жорстка математична регуляризація
        auto_class_weights='Balanced', # Даємо вагу меншості
        eval_metric='PRAUC',        # Орієнтуємось на площу під PR-кривою
        random_seed=42,
        verbose=0                   # Вимикаємо спам у консолі
    )
    
    # Створюємо пули даних для CatBoost
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # 4. Навчання з Early Stopping
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50 # Зупиняємось, якщо PR-AUC не росте 50 ітерацій
    )
    
    # 5. Оцінка на валідаційному фолді
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Оскільки ми використали class_weights='Balanced', ймовірності будуть зміщені.
    # Стандартний поріг 0.5 може не підійти. Для бізнесу можемо підібрати його окремо, 
    # але поки візьмемо стандартний прогноз:
    y_pred = model.predict(X_val)
    
    # Розрахунок метрик
    pr_auc = average_precision_score(y_val, y_pred_proba)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
    
    metrics['pr_auc'].append(pr_auc)
    metrics['precision'].append(prec)
    metrics['recall'].append(rec)
    metrics['f2_score'].append(f2)
    
    print(f"Fold {fold+1} | PR-AUC: {pr_auc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F2: {f2:.3f}")

# 6. Фінальні результати
print("\n" + "-"*40)
print(f"Середній PR-AUC:   {np.mean(metrics['pr_auc']):.3f} (±{np.std(metrics['pr_auc']):.3f})")
print(f"Середня Precision: {np.mean(metrics['precision']):.3f} (±{np.std(metrics['precision']):.3f})")
print(f"Середній Recall:   {np.mean(metrics['recall']):.3f} (±{np.std(metrics['recall']):.3f})")
print(f"Середній F2-score: {np.mean(metrics['f2_score']):.3f} (±{np.std(metrics['f2_score']):.3f})")

# 7. Аналіз важливості фічей (на останньому фолді для наочності)
feature_importances = model.get_feature_importance(train_pool)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nТоп-5 найважливіших фічей:")
print(importance_df.head(5).to_string(index=False))