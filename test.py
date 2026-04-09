import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# =====================================================
# 0. BASE DATA
# =====================================================

df_base = df.copy()

X_all = df_base[features_to_use].copy()
y_all = df_base[TARGET_NAME].copy()

# =====================================================
# 1. SPLIT (ОДИН ДЛЯ ВСЬОГО)
# =====================================================

y_clf_full = (y_all > 0).astype(int)

X_train, X_val, y_train, y_val, y_train_clf, y_val_clf = train_test_split(
    X_all, y_all, y_clf_full,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_clf_full
)

# =====================================================
# 2. CLASSIFICATION MODEL
# =====================================================

cat_cols = [c for c in X_train.columns if X_train[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

clf_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf_model.fit(
    X_train, y_train_clf,
    eval_set=[(X_val, y_val_clf)],
    callbacks=[lgb.early_stopping(50)]
)

# --- метрики класифікації
probs_val = clf_model.predict_proba(X_val)[:, 1]
preds_clf = (probs_val > 0.3).astype(int)

print("\n=== CLASSIFICATION METRICS ===")
print("ROC-AUC:", roc_auc_score(y_val_clf, probs_val))
print(classification_report(y_val_clf, preds_clf))

# =====================================================
# 3. REGRESSION DATASET (ТІЛЬКИ >0)
# =====================================================

df_train_reg = pd.concat([X_train, y_train], axis=1)
df_val_reg   = pd.concat([X_val, y_val], axis=1)

df_train_reg = df_train_reg[df_train_reg[TARGET_NAME] > 0]
df_val_reg   = df_val_reg[df_val_reg[TARGET_NAME] > 0]

# =====================================================
# 4. TARGET PREPROCESSING (ТІЛЬКИ ДЛЯ REG)
# =====================================================

def preprocess_target(df_):
    df_ = df_.copy()

    df_ = df_[df_[TARGET_NAME] > 20]

    lower = df_[TARGET_NAME].quantile(0.1)
    upper = df_[TARGET_NAME].quantile(0.975)

    df_[TARGET_NAME] = df_[TARGET_NAME].clip(lower=lower)
    df_ = df_[df_[TARGET_NAME] < 6000]

    return df_

df_train_reg = preprocess_target(df_train_reg)
df_val_reg   = preprocess_target(df_val_reg)

# =====================================================
# 5. BUCKET MODEL (ТВОЯ ЛОГІКА)
# =====================================================

bins = [-1, 350, 550, 750, 925, 1100, 2000, 4000, np.inf]

y_train_raw = df_train_reg[TARGET_NAME]
y_val_raw   = df_val_reg[TARGET_NAME]

y_train_binned = pd.cut(y_train_raw, bins=bins, labels=False)
y_val_binned   = pd.cut(y_val_raw, bins=bins, labels=False)

X_train_reg = df_train_reg[features_to_use].copy()
X_val_reg   = df_val_reg[features_to_use].copy()

# категорії
for c in cat_cols:
    if c in X_train_reg.columns:
        X_train_reg[c] = X_train_reg[c].astype("category")
        X_val_reg[c] = X_val_reg[c].astype("category")

# медіани бакетів
bucket_medians = y_train_raw.groupby(y_train_binned).median().to_dict()

print("\nМедіани бакетів:")
print(bucket_medians)

# модель
reg_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(bins)-1,
    n_estimators=1500,
    learning_rate=0.03,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg_model.fit(
    X_train_reg, y_train_binned,
    eval_set=[(X_val_reg, y_val_binned)],
    callbacks=[lgb.early_stopping(50)]
)

# =====================================================
# 6. REG METRICS
# =====================================================

probs_reg = reg_model.predict_proba(X_val_reg)

y_pred_reg = np.zeros(len(X_val_reg))
for i in range(len(bins)-1):
    y_pred_reg += probs_reg[:, i] * bucket_medians[i]

mae_reg = mean_absolute_error(y_val_raw, y_pred_reg)
print("\n=== REGRESSION METRICS ===")
print("MAE (E[Y]):", mae_reg)

# =====================================================
# 7. FINAL PIPELINE (CLF + REG)
# =====================================================

# classification probabilities
probs_all = clf_model.predict_proba(X_val)[:, 1]

threshold = 0.3
mask = probs_all > threshold

# фінальний предикт
y_pred_final = np.zeros(len(X_val))

# регресія тільки для відібраних
X_val_selected = X_val[mask]

if len(X_val_selected) > 0:
    probs_selected = reg_model.predict_proba(X_val_selected)

    y_pred_selected = np.zeros(len(X_val_selected))
    for i in range(len(bins)-1):
        y_pred_selected += probs_selected[:, i] * bucket_medians[i]

    y_pred_final[mask] = y_pred_selected

# =====================================================
# 8. FINAL METRIC (BUSINESS)
# =====================================================

mae_final = mean_absolute_error(y_val, y_pred_final)

print("\n=== FINAL PIPELINE METRIC ===")
print("FINAL MAE:", mae_final)