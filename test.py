import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    log_loss
)

# =========================
# DATA
# =========================

df = train_dataset.copy()

# target
target_col = 'SEGMENT'

# features
drop_cols = ['SEGMENT', 'MOBILEPHONE', 'CONTRAGENTID']
X = df.drop(columns=drop_cols)

# label encoding
le = LabelEncoder()
y = le.fit_transform(df[target_col])

print("Класи:", list(le.classes_))
print("Розподіл класів:")
print(df[target_col].value_counts())


cat_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

for col in cat_features:
    X[col] = X[col].fillna('Missing').astype(str)

num_features = [col for col in X.columns if col not in cat_features]

for col in num_features:
    X[col] = X[col].fillna(X[col].median())


n_splits = 5

skf = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

metrics = {
    'f1_macro': [],
    'f1_weighted': [],
    'precision_macro': [],
    'recall_macro': [],
    'logloss': []
}

oof_proba = np.zeros((len(X), len(le.classes_)))
oof_pred = np.zeros(len(X), dtype=int)

models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y[train_idx], y[val_idx]

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_features
    )

    val_pool = Pool(
        X_val,
        y_val,
        cat_features=cat_features
    )

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        iterations=500,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=20,
        random_seed=42,
        auto_class_weights='Balanced',
        verbose=0
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50
    )

    val_proba = model.predict_proba(X_val)
    val_pred = np.argmax(val_proba, axis=1)

    oof_proba[val_idx] = val_proba
    oof_pred[val_idx] = val_pred

    f1_macro = f1_score(y_val, val_pred, average='macro')
    f1_weighted = f1_score(y_val, val_pred, average='weighted')
    precision_macro = precision_score(y_val, val_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_val, val_pred, average='macro', zero_division=0)
    fold_logloss = log_loss(y_val, val_proba)

    metrics['f1_macro'].append(f1_macro)
    metrics['f1_weighted'].append(f1_weighted)
    metrics['precision_macro'].append(precision_macro)
    metrics['recall_macro'].append(recall_macro)
    metrics['logloss'].append(fold_logloss)

    models.append(model)

    print(
        f"Fold {fold} | "
        f"F1 macro: {f1_macro:.3f} | "
        f"F1 weighted: {f1_weighted:.3f} | "
        f"Precision macro: {precision_macro:.3f} | "
        f"Recall macro: {recall_macro:.3f} | "
        f"LogLoss: {fold_logloss:.3f}"
    )

print("\n" + "-" * 50)
print(f"Mean F1 macro:      {np.mean(metrics['f1_macro']):.3f} ± {np.std(metrics['f1_macro']):.3f}")
print(f"Mean F1 weighted:   {np.mean(metrics['f1_weighted']):.3f} ± {np.std(metrics['f1_weighted']):.3f}")
print(f"Mean Precision:     {np.mean(metrics['precision_macro']):.3f} ± {np.std(metrics['precision_macro']):.3f}")
print(f"Mean Recall:        {np.mean(metrics['recall_macro']):.3f} ± {np.std(metrics['recall_macro']):.3f}")
print(f"Mean LogLoss:       {np.mean(metrics['logloss']):.3f} ± {np.std(metrics['logloss']):.3f}") 
print(classification_report(
    y,
    oof_pred,
    target_names=le.classes_,
    zero_division=0
))







cm = confusion_matrix(y, oof_pred)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in le.classes_],
    columns=[f"pred_{c}" for c in le.classes_]
)

cm_df




proba_df = pd.DataFrame(
    oof_proba,
    columns=[f'proba_{cls}' for cls in le.classes_]
)

result_df = train_dataset[['CONTRAGENTID', 'MOBILEPHONE', 'SEGMENT']].copy()

result_df['pred_segment'] = le.inverse_transform(oof_pred)
result_df = pd.concat([result_df, proba_df], axis=1)

result_df['confidence'] = oof_proba.max(axis=1)

result_df.head()



def confidence_bucket(p):
    if p >= 0.80:
        return 'High confidence'
    elif p >= 0.60:
        return 'Medium confidence'
    else:
        return 'Low confidence'

result_df['confidence_bucket'] = result_df['confidence'].apply(confidence_bucket)

result_df[['SEGMENT', 'pred_segment', 'confidence', 'confidence_bucket']].head()




hnwi_class_idx = list(le.classes_).index('HNWI')

result_df['proba_HNWI_direct'] = oof_proba[:, hnwi_class_idx]

result_df.sort_values('proba_HNWI_direct', ascending=False).head(30)




final_pool = Pool(
    X,
    y,
    cat_features=cat_features
)

final_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    iterations=500,
    depth=3,
    learning_rate=0.03,
    l2_leaf_reg=20,
    random_seed=42,
    auto_class_weights='Balanced',
    verbose=100
)

final_model.fit(final_pool)









def predict_segments(new_df, model, le, cat_features, train_columns):
    X_new = new_df.copy()

    # залишаємо ті самі фічі, що були на train
    X_new = X_new[train_columns].copy()

    for col in cat_features:
        X_new[col] = X_new[col].fillna('Missing').astype(str)

    num_features = [col for col in X_new.columns if col not in cat_features]

    for col in num_features:
        X_new[col] = X_new[col].fillna(X_new[col].median())

    proba = model.predict_proba(X_new)
    pred = np.argmax(proba, axis=1)

    pred_labels = le.inverse_transform(pred)

    proba_df = pd.DataFrame(
        proba,
        columns=[f'proba_{cls}' for cls in le.classes_],
        index=new_df.index
    )

    result = new_df.copy()
    result['pred_segment'] = pred_labels
    result['confidence'] = proba.max(axis=1)

    result = pd.concat([result, proba_df], axis=1)

    return result

train_columns = X.columns.tolist()

predicted_df = predict_segments(
    new_df=external_dataset,
    model=final_model,
    le=le,
    cat_features=cat_features,
    train_columns=train_columns
)

predicted_df.head()








final_candidates = predicted_df.sort_values('proba_HNWI', ascending=False)

final_candidates[
    [
        'CONTRAGENTID',
        'MOBILEPHONE',
        'pred_segment',
        'confidence',
        'proba_HNWI',
        'proba_PREM',
        'proba_MASS',
        'price_usd',
        'BALANCE',
        'mark_group'
    ]
].head(100)




