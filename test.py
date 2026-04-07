import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import joblib

# 1. Формуємо таргет: 1 - є прибуток, 0 - немає (на повному df!)
y_clf = (df["CURR_ACC"] > 0.05).astype(int)
X_clf = df[features_to_use].copy()

# 2. Розбиття даних
X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

# Форматування категорій
cat_cols = [c for c in X_train_clf.columns if X_train_clf[c].dtype.name in ("object", "category")]
for c in cat_cols:
    X_train_clf[c] = X_train_clf[c].astype("category")
    X_val_clf[c] = X_val_clf[c].astype("category")

# 3. Ініціалізація та тренування класифікатора (з регуляризацією)
clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=20,
    max_depth=5,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.7,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train_clf, y_train_clf,
    eval_set=[(X_val_clf, y_val_clf)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

# 4. Пошук ОПТИМАЛЬНОГО порогу
y_pred_proba_clf = clf.predict_proba(X_val_clf)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val_clf, y_pred_proba_clf)
fscore = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
optimal_idx = np.argmax(fscore)
optimal_threshold = thresholds[optimal_idx]

print(f"ROC-AUC: {roc_auc_score(y_val_clf, y_pred_proba_clf):.4f}")
print(f"Знайдено оптимальний поріг: {optimal_threshold:.4f} (F1: {fscore[optimal_idx]:.4f})")




# Для інференсу беремо вибірку, де є і нулі, і прибутки
X_test_final = X_val_clf.copy()

# Отримуємо справжні значення з оригінального датафрейму (без логарифмів)
y_true_final = df.loc[X_test_final.index, "CURR_ACC"]

# ==========================================
# ЕТАП 1: Працює Класифікатор
# Використовуємо знайдений оптимальний поріг
# ==========================================
clf_data = joblib.load('hurdle_classifier.pkl')
clf_model = clf_data['model']
opt_thresh = clf_data['threshold']

proba = clf_model.predict_proba(X_test_final)[:, 1]
# Якщо ймовірність вища за поріг - клієнт прибутковий (1), інакше (0)
is_profitable = (proba >= opt_thresh).astype(int) 

# ==========================================
# ЕТАП 2: Працює Регресор
# ==========================================
reg_model = joblib.load('hurdle_regressor.pkl')

# Прогнозуємо суму для всіх (матричні операції швидкі)
log_profit = reg_model.predict(X_test_final)
actual_profit = np.expm1(log_profit)

# ==========================================
# ЕТАП 3: Математичне злиття
# E[Y] = P(Y>0) * E[Y|Y>0]
# ==========================================
final_predicted_profit = is_profitable * actual_profit

# ==========================================
# Формуємо DataFrame для вашого Excel-звіту
# ==========================================
from sklearn.metrics import mean_absolute_error, median_absolute_error

validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_test_final.index,
    'True_Value': y_true_final,
    'Predicted': final_predicted_profit
})

# Рахуємо загальні фінансові метрики
mae = mean_absolute_error(validation_results['True_Value'], validation_results['Predicted'])
medae = median_absolute_error(validation_results['True_Value'], validation_results['Predicted'])

print("=" * 60)
print(f"ФІНАЛЬНІ МЕТРИКИ ДВОКОМПОНЕНТНОЇ МОДЕЛІ")
print(f"MAE: {mae:,.2f}")
print(f"MedAE: {medae:,.2f}")
print("=" * 60)

# ДАЛІ:
# df = validation_results.copy()
# ... ваш код генерації Excel (xlsxwriter) ...



