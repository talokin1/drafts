from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, fbeta_score, average_precision_score

# Готуємо X та y
X = df_client.drop(columns=['MOBILEPHONE', 'is_hnwi'])
y = df_client['is_hnwi'].astype(int)

# Визначаємо категоріальні фічі для CatBoost
cat_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
X[cat_features] = X[cat_features].fillna('Missing').astype(str)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

metrics = {'pr_auc': [], 'precision': [], 'recall': [], 'f2_score': [], 'best_thresh': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # Ініціалізація моделі з Focal Loss
    # alpha=0.7 (акцент на HNWI), gamma=2.0 (стандартне фокусування)
    model = CatBoostClassifier(
        iterations=300,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=20,
        loss_function='Focal:alpha=0.7;gamma=2.0', 
        eval_metric='PRAUC',
        random_seed=42,
        verbose=0
    )
    
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    
    # Прогнозуємо ЙМОВІРНОСТІ
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Оптимізація порогу під F2-score
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
    # Формула F2
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Фінальний прогноз за знайденим порогом
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    
    # Рахуємо метрики
    pr_auc = average_precision_score(y_val, y_pred_proba)
    prec = precision_score(y_val, y_pred_optimal, zero_division=0)
    rec = recall_score(y_val, y_pred_optimal, zero_division=0)
    f2 = fbeta_score(y_val, y_pred_optimal, beta=2, zero_division=0)
    
    metrics['pr_auc'].append(pr_auc)
    metrics['precision'].append(prec)
    metrics['recall'].append(rec)
    metrics['f2_score'].append(f2)
    metrics['best_thresh'].append(best_threshold)
    
    print(f"Fold {fold+1} | Thresh: {best_threshold:.3f} | PR-AUC: {pr_auc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F2: {f2:.3f}")

print("\n" + "-"*40)
print(f"Середній PR-AUC:   {np.mean(metrics['pr_auc']):.3f} (±{np.std(metrics['pr_auc']):.3f})")
print(f"Середня Precision: {np.mean(metrics['precision']):.3f} (±{np.std(metrics['precision']):.3f})")
print(f"Середній Recall:   {np.mean(metrics['recall']):.3f} (±{np.std(metrics['recall']):.3f})")
print(f"Середній F2-score: {np.mean(metrics['f2_score']):.3f} (±{np.std(metrics['f2_score']):.3f})")

# Feature Importance на останньому фолді
feature_importances = model.get_feature_importance(train_pool)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nТоп-5 фічей:")
print(importance_df.head(5).to_string(index=False))