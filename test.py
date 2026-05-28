all_classes = set(np.unique(y))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y[train_idx], y[val_idx]

    if set(np.unique(y_train)) != all_classes:
        print(f"Fold {fold} skipped: train has classes {np.unique(y_train)}")
        continue

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        iterations=300,
        depth=2,
        learning_rate=0.03,
        l2_leaf_reg=30,
        random_seed=42,
        auto_class_weights='Balanced',
        verbose=0
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=30
    )

    val_proba = model.predict_proba(X_val)
    val_pred = np.argmax(val_proba, axis=1)

    oof_proba[val_idx] = val_proba
    oof_pred[val_idx] = val_pred

    print(f"\nFold {fold}")
    print(classification_report(
        y_val,
        val_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    models.append(model)







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









hnwi_idx = list(le.classes_).index('HNWI')

result_df = train_dataset[['CONTRAGENTID', 'MOBILEPHONE', 'SEGMENT']].copy()

for i, cls in enumerate(le.classes_):
    result_df[f'proba_{cls}'] = oof_proba[:, i]

result_df['pred_segment'] = le.inverse_transform(oof_pred)
result_df['confidence'] = oof_proba.max(axis=1)

result_df = result_df.sort_values('proba_HNWI', ascending=False)

result_df






def precision_recall_at_k(df, proba_col, target_col, positive_label, k):
    top_k = df.sort_values(proba_col, ascending=False).head(k)

    precision_at_k = (top_k[target_col] == positive_label).mean()
    recall_at_k = (top_k[target_col] == positive_label).sum() / (df[target_col] == positive_label).sum()

    return precision_at_k, recall_at_k


for k in [20, 50, 100, 200, 300]:
    p_at_k, r_at_k = precision_recall_at_k(
        result_df,
        proba_col='proba_HNWI',
        target_col='SEGMENT',
        positive_label='HNWI',
        k=k
    )

    print(f"Top-{k}: Precision@K={p_at_k:.3f}, Recall@K={r_at_k:.3f}")





baseline_df = train_dataset.copy()

baseline_df = baseline_df.sort_values('price_usd', ascending=False)

for k in [20, 50, 100, 200, 300]:
    top_k = baseline_df.head(k)

    precision_at_k = (top_k['SEGMENT'] == 'HNWI').mean()
    recall_at_k = (top_k['SEGMENT'] == 'HNWI').sum() / (baseline_df['SEGMENT'] == 'HNWI').sum()

    print(f"price_usd Top-{k}: Precision@K={precision_at_k:.3f}, Recall@K={recall_at_k:.3f}")





hnwi_result = train_dataset[['CONTRAGENTID', 'MOBILEPHONE', 'SEGMENT']].copy()

hnwi_result['proba_HNWI'] = oof_hnwi_proba
hnwi_result['rank_HNWI'] = hnwi_result['proba_HNWI'].rank(
    ascending=False,
    method='first'
)

hnwi_result = hnwi_result.sort_values('proba_HNWI', ascending=False)

hnwi_result.head(100)