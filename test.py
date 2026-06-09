import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

proba_col = "PRIMARY_LIABS"   # або твоя колонка з probability
target_col = "LIABS_ACTIVE"   # факт: були пасиви чи ні

thresholds = np.arange(0.05, 0.55, 0.05)

rows = []

for thr in thresholds:
    y_pred = (df_val[proba_col] >= thr).astype(int)
    y_true = df_val[target_col].astype(int)

    rows.append({
        "threshold": thr,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "predicted_positive_count": y_pred.sum(),
        "predicted_positive_share": y_pred.mean()
    })

threshold_report = pd.DataFrame(rows)

threshold_report



import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))

plt.plot(threshold_report["threshold"], threshold_report["precision"], marker="o", label="Precision")
plt.plot(threshold_report["threshold"], threshold_report["recall"], marker="o", label="Recall")
plt.plot(threshold_report["threshold"], threshold_report["f1"], marker="o", label="F1")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Liabilities propensity threshold analysis")
plt.legend()
plt.grid(True)
plt.show()








proba_col = "PRIMARY_LIABS"
raw_col = "LIABILITIES_POTENTIAL_RAW"
top_bucket_col = "LIABS_TOP_BUCKET"

thresholds = np.arange(0.05, 0.55, 0.05)

rows = []

for thr in thresholds:
    zero_mask = (
        (df_adj[raw_col] != 0)
        & (df_adj[proba_col] < thr)
        & (df_adj[top_bucket_col] == "0")
    )

    discount_mask = (
        (df_adj[raw_col] != 0)
        & (df_adj[proba_col] < thr)
        & (df_adj[top_bucket_col] == "0-100K")
    )

    rows.append({
        "threshold": thr,
        "zeroed_clients": zero_mask.sum(),
        "zeroed_share_from_raw_positive": zero_mask.sum() / (df_adj[raw_col] != 0).sum(),
        "zeroed_raw_income_sum": df_adj.loc[zero_mask, raw_col].sum(),
        "discounted_0_100k_clients": discount_mask.sum(),
        "discounted_0_100k_raw_income_sum": df_adj.loc[discount_mask, raw_col].sum()
    })

business_threshold_report = pd.DataFrame(rows)

business_threshold_report