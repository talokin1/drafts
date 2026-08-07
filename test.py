import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Підбір optimal threshold по OOF predictions
# --------------------------------------------------

precision, recall, thresholds = precision_recall_curve(y_all, oof_proba)

f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)

best_idx = np.argmax(f1)
BEST_THRESHOLD = thresholds[best_idx]

print(f'Best threshold: {BEST_THRESHOLD:.4f}')
print(f'Precision:      {precision[best_idx]:.4f}')
print(f'Recall:         {recall[best_idx]:.4f}')
print(f'F1:             {f1[best_idx]:.4f}')




oof_pred = (oof_proba >= BEST_THRESHOLD).astype(int)

metrics = pd.Series({
    'Precision': precision_score(y_all, oof_pred),
    'Recall': recall_score(y_all, oof_pred),
    'F1': f1_score(y_all, oof_pred),
    'ROC_AUC': roc_auc_score(y_all, oof_proba)
})

display(metrics)

cm = pd.DataFrame(
    confusion_matrix(y_all, oof_pred),
    index=['Actual Not Golden', 'Actual Golden'],
    columns=['Pred Not Golden', 'Pred Golden']
)

display(cm)


score_df['GOLDEN_SCORE'] = oof_proba
score_df['MODEL_CLASS'] = np.where(score_df['GOLDEN_SCORE'] >= BEST_THRESHOLD, 'Golden', 'Not Golden')

inference_result = (
    score_df.loc[
        score_df[TARGET].eq(0),
        ['CONTRAGENTID', 'GOLDEN_SCORE', 'MODEL_CLASS']
    ]
    .sort_values('GOLDEN_SCORE', ascending=False)
    .reset_index(drop=True)
)

inference_result['GOLDEN_PROPENSITY_PCT'] = (inference_result['GOLDEN_SCORE'] * 100).round(2)
inference_result['GOLDEN_RANK'] = np.arange(1, len(inference_result) + 1)

inference_result.head(50)