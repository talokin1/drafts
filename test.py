import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. СТВОРЕННЯ ТАРГЕТУ (Біннінг)
# ---------------------------------------------------------

# ВАРІАНТ А: Автоматичний (3 рівні класи по 33% даних)
# Це найкраще для старту, щоб перевірити здатність моделі розділяти дані
# df['target_class'] = pd.qcut(df['CURR_ACC'], q=3, labels=[0, 1, 2])

# ВАРІАНТ Б: Твій бізнес-підхід (ручні пороги)
# Треба перевірити, чи не буде перекосу класів.
def create_bins(x):
    if x <= 100:
        return 0 # Low / Loss
    elif x <= 5000:
        return 1 # Medium
    else:
        return 2 # High

# Використовуємо оригінальну колонку (не логарифм!)
df['target_class'] = df['CURR_ACC'].apply(create_bins)

print("Розподіл класів:")
print(df['target_class'].value_counts(normalize=True))

# ---------------------------------------------------------
# 2. ПІДГОТОВКА ДАНИХ
# ---------------------------------------------------------
# Видаляємо ліки даних (таргет і його похідні)
drop_cols = ['CURR_ACC', 'target_class'] 
# Якщо є логарифми таргета, їх теж треба прибрати, бо це лік!
# наприклад: drop_cols += ['CURR_ACC_log']

X = df.drop(columns=drop_cols, errors='ignore')
y = df['target_class']

# Розбиття (Stratified! щоб зберегти пропорції класів)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 3. НАВЧАННЯ (LGBMClassifier)
# ---------------------------------------------------------
clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,  # Кількість класів
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced' # Допомагає, якщо класи нерівні
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss', # або 'multi_error'
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
)

# ---------------------------------------------------------
# 4. РЕЗУЛЬТАТИ
# ---------------------------------------------------------
y_pred = clf.predict(X_val)

print("="*60)
print("ACCURACY:", accuracy_score(y_val, y_pred))
print("="*60)
print(classification_report(y_val, y_pred, target_names=['Low', 'Medium', 'High']))

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred Low', 'Pred Med', 'Pred High'],
            yticklabels=['True Low', 'True Med', 'True High'])
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()