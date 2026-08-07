import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

TARGET = 'GOLDEN_TARGET'
ID_COL = 'CONTRAGENTID'
RANDOM_STATE = 42

FEATURE_COLS = X_train.columns.tolist()
CAT_COLS = [col for col in categorical_cols if col in FEATURE_COLS]


def prepare_features(df):
    X = df.reindex(columns=FEATURE_COLS).copy()

    for col in CAT_COLS:
        X[col] = X[col].astype('string').fillna('__MISSING__').astype(str)

    numeric_cols = X.columns.difference(CAT_COLS)
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return X


def get_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)

    return pd.Series({
        'Precision': precision_score(y_true, pred, zero_division=0),
        'Recall': recall_score(y_true, pred, zero_division=0),
        'F1': f1_score(y_true, pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, proba)
    })


score_df = modeling_df.loc[modeling_df[TARGET].notna()].copy().reset_index(drop=True)

assert score_df[ID_COL].is_unique, 'Є дублікати CONTRAGENTID — тоді треба використовувати group split'

X_all = prepare_features(score_df)
y_all = score_df[TARGET].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_proba = np.zeros(len(score_df))


base_params = model.get_params()
base_params['iterations'] = model.tree_count_
base_params['verbose'] = False
base_params['allow_writing_files'] = False

for fold, (train_idx, val_idx) in enumerate(cv.split(X_all, y_all), 1):
    X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

    train_pool_fold = Pool(X_tr, y_tr, cat_features=CAT_COLS)
    val_pool_fold = Pool(X_val, y_val, cat_features=CAT_COLS)

    fold_params = base_params.copy()
    fold_params['random_seed'] = RANDOM_STATE + fold

    fold_model = CatBoostClassifier(**fold_params)
    fold_model.fit(train_pool_fold)

    oof_proba[val_idx] = fold_model.predict_proba(val_pool_fold)[:, 1]


metrics = get_metrics(y_all, oof_proba)

print('OOF METRICS:')
display(metrics)

pred = (oof_proba >= 0.5).astype(int)

cm = pd.DataFrame(
    confusion_matrix(y_all, pred),
    index=['Actual NOT Golden', 'Actual Golden'],
    columns=['Pred NOT Golden', 'Pred Golden']
)

display(cm)


score_df['GOLDEN_SCORE'] = oof_proba

inference_result = (
    score_df.loc[
        score_df[TARGET].eq(0),
        [ID_COL, 'GOLDEN_SCORE']
    ]
    .sort_values('GOLDEN_SCORE', ascending=False)
    .reset_index(drop=True)
)

inference_result['GOLDEN_RANK'] = np.arange(1, len(inference_result) + 1)
inference_result['HIGH_GOLDEN_PROPENSITY'] = (inference_result['GOLDEN_SCORE'] >= 0.5).astype('int8')

display(inference_result.head(50))