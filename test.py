import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from catboost import CatBoostClassifier, Pool

RANDOM_STATE = 42
TARGET = 'GOLDEN_TARGET'
ID_COLS = ['CONTRAGENTID']



def dataset_audit(df, target=None, corr_threshold=0.95, near_constant_threshold=0.995):
    X = df.drop(columns=[target], errors='ignore').copy()
    
    missing_pct = X.isna().mean()
    nunique = X.nunique(dropna=False)
    unique_ratio = nunique / len(X)
    top_value_share = X.apply(lambda col: col.value_counts(normalize=True, dropna=False).iloc[0])
    
    feature_report = pd.DataFrame({
        'dtype': X.dtypes.astype(str),
        'missing_pct': missing_pct,
        'nunique': nunique,
        'unique_ratio': unique_ratio,
        'top_value_share': top_value_share
    })
    
    feature_report['is_constant'] = feature_report['nunique'].le(1)
    feature_report['is_near_constant'] = feature_report['top_value_share'].ge(near_constant_threshold)
    feature_report['potential_id'] = feature_report['unique_ratio'].ge(0.98)
    
    duplicate_columns = X.columns[X.T.duplicated()].tolist()
    
    numeric_cols = X.select_dtypes(include=np.number).columns
    numeric_data = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    if len(numeric_cols) > 1:
        corr = numeric_data.corr(method='spearman').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_pairs = upper.stack().reset_index()
        high_corr_pairs.columns = ['feature_1', 'feature_2', 'abs_spearman']
        high_corr_pairs = high_corr_pairs.query('abs_spearman >= @corr_threshold').sort_values('abs_spearman', ascending=False)
    else:
        high_corr_pairs = pd.DataFrame(columns=['feature_1', 'feature_2', 'abs_spearman'])
    
    summary = {
        'rows': len(df),
        'columns': df.shape[1],
        'duplicated_rows': int(df.duplicated().sum()),
        'duplicated_column_names': int(df.columns.duplicated().sum()),
        'constant_columns': feature_report.index[feature_report['is_constant']].tolist(),
        'near_constant_columns': feature_report.index[feature_report['is_near_constant']].tolist(),
        'high_missing_columns': feature_report.index[feature_report['missing_pct'].ge(0.95)].tolist(),
        'potential_id_columns': feature_report.index[feature_report['potential_id']].tolist(),
        'duplicate_columns': duplicate_columns
    }
    
    return summary, feature_report.sort_values('missing_pct', ascending=False), high_corr_pairs

audit_summary, feature_report, high_corr_pairs = dataset_audit(clients, target=TARGET)

audit_summary









def calculate_vif(df, columns, max_features=80):
    columns = [col for col in columns if col in df.columns]
    
    if len(columns) > max_features:
        raise ValueError(f'Для VIF передано {len(columns)} ознак. Спочатку скороти список до {max_features}.')
    
    X = df[columns].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    X = X.loc[:, X.nunique().gt(1)]
    
    std = X.std(ddof=0)
    X = X.loc[:, std.gt(0)]
    X = (X - X.mean()) / X.std(ddof=0)
    
    result = []
    
    for i, column in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.inf
        
        result.append({'feature': column, 'VIF': vif})
    
    return pd.DataFrame(result).sort_values('VIF', ascending=False)

numeric_candidates = clients.select_dtypes(include=np.number).columns.difference([TARGET, *ID_COLS]).tolist()
vif_report = calculate_vif(clients, numeric_candidates[:80])
vif_report.head(30)












DIRECT_LEAKAGE_COLS = [
    'PACKAGE',
    'TOTAL_PORTFOLIO',
    'LIABILITIES_UAH',
    'INCOME(COM+INTEREST)',
    'AMT_DEB_CARD',
    
    'PACKAGE_FLAG',
    'TOTAL_PORTFOLIO_FLAG',
    'AUM_UAH_FLAG',
    'INCOME_FLAG',
    'POS_FLAG',
    
    'GOLDEN',
    'OTHER_GOLDEN_COUNT',
    'GOLDEN_CRITERIA_COUNT'
]

modeling_df = clients.dropna(subset=[TARGET]).copy()
modeling_df[TARGET] = modeling_df[TARGET].astype('int8')

drop_before_model = [TARGET, *ID_COLS, *DIRECT_LEAKAGE_COLS]

X = modeling_df.drop(columns=drop_before_model, errors='ignore')
y = modeling_df[TARGET]
target_distribution = pd.DataFrame({
    'count': y.value_counts(),
    'share': y.value_counts(normalize=True)
})

target_distribution





X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

pd.DataFrame({
    'dev': y_dev.value_counts(normalize=True),
    'test': y_test.value_counts(normalize=True)
})








def numeric_split_report(
    X_train,
    X_test,
    alpha=0.05,
    ks_limit=0.10,
    rank_biserial_limit=0.10,
    smd_limit=0.10,
    missing_diff_limit=0.05
):
    numeric_cols = X_train.select_dtypes(include=np.number).columns.intersection(X_test.columns)
    rows = []
    
    for col in numeric_cols:
        train_values = pd.to_numeric(X_train[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        test_values = pd.to_numeric(X_test[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        
        if min(len(train_values), len(test_values)) < 20:
            continue
        
        ks_result = ks_2samp(train_values, test_values, alternative='two-sided', method='auto')
        mw_result = mannwhitneyu(train_values, test_values, alternative='two-sided', method='asymptotic')
        
        pooled_std = np.sqrt((train_values.var(ddof=1) + test_values.var(ddof=1)) / 2)
        smd = 0.0 if pooled_std == 0 else (train_values.mean() - test_values.mean()) / pooled_std
        rank_biserial = 2 * mw_result.statistic / (len(train_values) * len(test_values)) - 1
        
        rows.append({
            'feature': col,
            'ks_stat': ks_result.statistic,
            'ks_pvalue': ks_result.pvalue,
            'mw_pvalue': mw_result.pvalue,
            'rank_biserial': rank_biserial,
            'smd': smd,
            'missing_diff': abs(X_train[col].isna().mean() - X_test[col].isna().mean())
        })
    
    report = pd.DataFrame(rows)
    
    if report.empty:
        return report
    
    report['ks_pvalue_fdr'] = multipletests(report['ks_pvalue'], method='fdr_bh')[1]
    report['mw_pvalue_fdr'] = multipletests(report['mw_pvalue'], method='fdr_bh')[1]
    
    report['shift_flag'] = (
        ((report['ks_pvalue_fdr'] < alpha) & (report['ks_stat'] > ks_limit)) |
        ((report['mw_pvalue_fdr'] < alpha) & (report['rank_biserial'].abs() > rank_biserial_limit)) |
        (report['smd'].abs() > smd_limit) |
        (report['missing_diff'] > missing_diff_limit)
    )
    
    return report.sort_values(
        ['shift_flag', 'ks_stat', 'smd'],
        ascending=[False, False, False]
    )


def categorical_split_report(X_train, X_test, tv_limit=0.10, max_share_diff_limit=0.05):
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'string', 'bool']).columns
    rows = []
    
    for col in categorical_cols:
        train_dist = X_train[col].astype('string').fillna('__MISSING__').value_counts(normalize=True)
        test_dist = X_test[col].astype('string').fillna('__MISSING__').value_counts(normalize=True)
        
        categories = train_dist.index.union(test_dist.index)
        train_dist = train_dist.reindex(categories, fill_value=0)
        test_dist = test_dist.reindex(categories, fill_value=0)
        
        differences = (train_dist - test_dist).abs()
        
        rows.append({
            'feature': col,
            'n_categories': len(categories),
            'tv_distance': 0.5 * differences.sum(),
            'max_share_diff': differences.max()
        })
    
    report = pd.DataFrame(rows)
    
    if report.empty:
        return report
    
    report['shift_flag'] = (
        (report['tv_distance'] > tv_limit) |
        (report['max_share_diff'] > max_share_diff_limit)
    )
    
    return report.sort_values(
        ['shift_flag', 'tv_distance'],
        ascending=[False, False]
    )

numeric_split = numeric_split_report(X_dev, X_test)
categorical_split = categorical_split_report(X_dev, X_test)



numeric_failed_share = numeric_split['shift_flag'].mean() if len(numeric_split) else 0
categorical_failed_share = categorical_split['shift_flag'].mean() if len(categorical_split) else 0
target_rate_diff = abs(y_dev.mean() - y_test.mean())

split_summary = pd.Series({
    'dev_golden_rate': y_dev.mean(),
    'test_golden_rate': y_test.mean(),
    'target_rate_diff': target_rate_diff,
    'numeric_failed_share': numeric_failed_share,
    'categorical_failed_share': categorical_failed_share,
    'split_ok': (
        target_rate_diff <= 0.01 and
        numeric_failed_share <= 0.05 and
        categorical_failed_share <= 0.05
    )
})

split_summary







def fit_cleaning_rules(
    X_train,
    missing_threshold=0.99,
    near_constant_threshold=0.999,
    drop_near_constant=False
):
    missing_pct = X_train.isna().mean()
    nunique = X_train.nunique(dropna=False)
    top_share = X_train.apply(lambda col: col.value_counts(normalize=True, dropna=False).iloc[0])
    
    high_missing = missing_pct[missing_pct >= missing_threshold].index.tolist()
    constant = nunique[nunique <= 1].index.tolist()
    near_constant = top_share[top_share >= near_constant_threshold].index.tolist()
    duplicate = X_train.columns[X_train.T.duplicated()].tolist()
    
    drop_cols = set(high_missing + constant + duplicate)
    
    if drop_near_constant:
        drop_cols.update(near_constant)
    
    return {
        'drop_cols': sorted(drop_cols),
        'high_missing_cols': high_missing,
        'constant_cols': constant,
        'near_constant_cols': near_constant,
        'duplicate_cols': duplicate
    }


X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_STATE
)
cleaning_rules = fit_cleaning_rules(X_train_raw)
cleaning_rules


X_train = X_train_raw.drop(columns=cleaning_rules['drop_cols'], errors='ignore').copy()
X_valid = X_valid_raw.reindex(columns=X_train.columns).copy()
X_test_clean = X_test.reindex(columns=X_train.columns).copy()





def infer_categorical_columns(X, manual_categorical=None):
    manual_categorical = manual_categorical or []
    
    dtype_categorical = X.select_dtypes(
        include=['object', 'category', 'string', 'bool']
    ).columns.tolist()
    
    semantic_categorical = [
        col for col in X.columns
        if col.upper().endswith(('_ID', '_CODE'))
        or col.upper() in {'GENDER', 'SEGMENT'}
    ]
    
    return sorted(set(dtype_categorical + semantic_categorical + manual_categorical).intersection(X.columns))

categorical_cols = infer_categorical_columns(X_train)
len(categorical_cols), categorical_cols[:30]





def prepare_catboost_frames(X_train, X_valid, X_test, categorical_cols):
    train = X_train.copy()
    valid = X_valid.copy()
    test = X_test.copy()
    
    for frame in [train, valid, test]:
        for col in categorical_cols:
            frame[col] = frame[col].astype('string').fillna('__MISSING__').astype(str)
        
        numeric_cols = frame.columns.difference(categorical_cols)
        frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    return train, valid, test

X_train, X_valid, X_test_clean = prepare_catboost_frames(
    X_train,
    X_valid,
    X_test_clean,
    categorical_cols
)









train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=categorical_cols
)

valid_pool = Pool(
    data=X_valid,
    label=y_valid,
    cat_features=categorical_cols
)

test_pool = Pool(
    data=X_test_clean,
    label=y_test,
    cat_features=categorical_cols
)
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=7,
    random_strength=1,
    bootstrap_type='Bernoulli',
    subsample=0.80,
    loss_function='Logloss',
    eval_metric='PRAUC:type=Classic',
    random_seed=RANDOM_STATE,
    allow_writing_files=False,
    verbose=100
)

model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True,
    early_stopping_rounds=150
)
def evaluate_binary_model(y_true, probabilities, threshold=0.50, top_fraction=0.10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    
    top_k = max(1, int(np.ceil(len(y_true) * top_fraction)))
    top_indices = np.argsort(-probabilities)[:top_k]
    top_precision = y_true[top_indices].mean()
    top_recall = y_true[top_indices].sum() / max(y_true.sum(), 1)
    base_rate = y_true.mean()
    
    return pd.Series({
        'ROC_AUC': roc_auc_score(y_true, probabilities),
        'Average_Precision': average_precision_score(y_true, probabilities),
        'Precision_0.5': precision_score(y_true, predictions, zero_division=0),
        'Recall_0.5': recall_score(y_true, predictions, zero_division=0),
        'F1_0.5': f1_score(y_true, predictions, zero_division=0),
        f'Precision_top_{int(top_fraction * 100)}pct': top_precision,
        f'Recall_top_{int(top_fraction * 100)}pct': top_recall,
        f'Lift_top_{int(top_fraction * 100)}pct': top_precision / base_rate if base_rate else np.nan,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    })

test_proba = model.predict_proba(test_pool)[:, 1]
test_metrics = evaluate_binary_model(y_test, test_proba)

test_metrics











test_result = modeling_df.loc[X_test_clean.index, ['CONTRAGENTID']].copy()
test_result['GOLDEN_TARGET'] = y_test
test_result['GOLDEN_SCORE'] = test_proba
test_result = test_result.sort_values('GOLDEN_SCORE', ascending=False)

test_result.head(30)



feature_importance = model.get_feature_importance(
    type='PredictionValuesChange',
    prettified=True
)

feature_importance.head(50)