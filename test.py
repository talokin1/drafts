import numpy as np
import pandas as pd

# =========================
# 1. Налаштування
# =========================

INPUT_FILE = "your_file.xlsx"        # заміни на назву свого файлу
SHEET_NAME = "Model Accuracy"

OUTPUT_FILE = "peter_classification_metrics.xlsx"

FACT_COL = "MONTHLY_INCOME"
PRED_COL = "POTENTIAL_INCOME"

bins = [-np.inf, 300, 1000, 2500, 10_000, np.inf]
labels = [
    "total hopeless",
    "bad quality",
    "grey zone",
    "green zone",
    "fantastic"
]

# =========================
# 2. Зчитування файлу
# =========================

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# прибираємо повністю пусті рядки
df = df.dropna(how="all").copy()

# приводимо income-колонки до числового формату
for col in [FACT_COL, PRED_COL]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace("-", np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# залишаємо тільки рядки, де є факт і прогноз
eval_df = df.dropna(subset=[FACT_COL, PRED_COL]).copy()

# якщо треба оцінювати тільки клієнтів з фактичним income != 0
eval_df = eval_df[eval_df[FACT_COL] != 0].copy()

print(f"Кількість клієнтів для оцінки: {len(eval_df):,}")

# =========================
# 3. Peter-класи
# =========================

eval_df["PETER_FACT"] = pd.cut(
    eval_df[FACT_COL],
    bins=bins,
    labels=labels,
    right=False
)

eval_df["PETER_PRED"] = pd.cut(
    eval_df[PRED_COL],
    bins=bins,
    labels=labels,
    right=False
)

# =========================
# 4. Confusion Matrix
# =========================

conf_matrix = pd.crosstab(
    eval_df["PETER_FACT"],
    eval_df["PETER_PRED"],
    rownames=["Fact"],
    colnames=["Prediction"],
    dropna=False
)

# гарантуємо правильний порядок класів
conf_matrix = conf_matrix.reindex(index=labels, columns=labels, fill_value=0)

conf_matrix_with_total = conf_matrix.copy()
conf_matrix_with_total["Grand Total"] = conf_matrix_with_total.sum(axis=1)
conf_matrix_with_total.loc["Grand Total"] = conf_matrix_with_total.sum(axis=0)

print("\n=== Confusion Matrix ===")
display(conf_matrix_with_total)

# =========================
# 5. Confusion Matrix у %
# =========================

conf_matrix_pct_total = conf_matrix / conf_matrix.values.sum()

conf_matrix_pct_total_with_total = conf_matrix_pct_total.copy()
conf_matrix_pct_total_with_total["Grand Total"] = conf_matrix_pct_total_with_total.sum(axis=1)
conf_matrix_pct_total_with_total.loc["Grand Total"] = conf_matrix_pct_total_with_total.sum(axis=0)

print("\n=== Confusion Matrix, % від усієї бази ===")
display((conf_matrix_pct_total_with_total * 100).round(2))

# =========================
# 6. Точність по кожному класу
# =========================

class_metrics = []

for cls in labels:
    fact_count = conf_matrix.loc[cls].sum()
    correct_count = conf_matrix.loc[cls, cls]
    
    if fact_count > 0:
        class_accuracy = correct_count / fact_count
    else:
        class_accuracy = np.nan
    
    class_metrics.append({
        "PETER_CLASS": cls,
        "FACT_COUNT": fact_count,
        "CORRECT_PRED_COUNT": correct_count,
        "CLASS_ACCURACY": class_accuracy
    })

class_metrics_df = pd.DataFrame(class_metrics)

class_metrics_df["CLASS_ACCURACY_%"] = (
    class_metrics_df["CLASS_ACCURACY"] * 100
).round(2)

print("\n=== Accuracy by Peter class ===")
display(class_metrics_df)

# =========================
# 7. Загальна точність
# =========================

total_count = len(eval_df)
correct_count = (eval_df["PETER_FACT"] == eval_df["PETER_PRED"]).sum()
overall_accuracy = correct_count / total_count

# =========================
# 8. Свій або сусідній клас
# =========================

class_to_rank = {
    "total hopeless": 0,
    "bad quality": 1,
    "grey zone": 2,
    "green zone": 3,
    "fantastic": 4
}

eval_df["FACT_RANK"] = eval_df["PETER_FACT"].map(class_to_rank).astype(int)
eval_df["PRED_RANK"] = eval_df["PETER_PRED"].map(class_to_rank).astype(int)

eval_df["CLASS_DISTANCE"] = (
    eval_df["FACT_RANK"] - eval_df["PRED_RANK"]
).abs()

same_or_neighbor_count = (eval_df["CLASS_DISTANCE"] <= 1).sum()
same_or_neighbor_accuracy = same_or_neighbor_count / total_count

# =========================
# 9. Додаткові бізнес-метрики
# =========================

summary_metrics = pd.DataFrame({
    "Metric": [
        "Valid Clients Count",
        "Correct Peter Class Count",
        "Overall Peter Class Accuracy",
        "Same or Neighbor Class Count",
        "Same or Neighbor Class Accuracy",
        "MAE Non-Zero Clients"
    ],
    "Value": [
        total_count,
        correct_count,
        overall_accuracy,
        same_or_neighbor_count,
        same_or_neighbor_accuracy,
        np.mean(np.abs(eval_df[FACT_COL] - eval_df[PRED_COL]))
    ]
})

summary_metrics["Value_for_slide"] = summary_metrics["Value"]

summary_metrics.loc[
    summary_metrics["Metric"].str.contains("Accuracy"),
    "Value_for_slide"
] = (
    summary_metrics.loc[
        summary_metrics["Metric"].str.contains("Accuracy"),
        "Value"
    ] * 100
).round(2).astype(str) + "%"

summary_metrics.loc[
    summary_metrics["Metric"].isin([
        "Valid Clients Count",
        "Correct Peter Class Count",
        "Same or Neighbor Class Count",
        "MAE Non-Zero Clients"
    ]),
    "Value_for_slide"
] = (
    summary_metrics.loc[
        summary_metrics["Metric"].isin([
            "Valid Clients Count",
            "Correct Peter Class Count",
            "Same or Neighbor Class Count",
            "MAE Non-Zero Clients"
        ]),
        "Value"
    ].round(0).astype(int)
)

print("\n=== Summary Metrics ===")
display(summary_metrics)

# =========================
# 10. Таблиця для слайду
# =========================

slide_metrics = pd.DataFrame({
    "Metric": [
        "Валідаційна база",
        "Точний збіг Peter-класу",
        "Збіг у своєму або сусідньому класі",
        "MAE для non-zero clients"
    ],
    "Value": [
        f"{total_count:,}".replace(",", " "),
        f"{overall_accuracy * 100:.1f}% ({correct_count:,} клієнтів)".replace(",", " "),
        f"{same_or_neighbor_accuracy * 100:.1f}% ({same_or_neighbor_count:,} клієнтів)".replace(",", " "),
        f"{np.mean(np.abs(eval_df[FACT_COL] - eval_df[PRED_COL])):,.0f}".replace(",", " ")
    ]
})

print("\n=== Metrics for Slide ===")
display(slide_metrics)

# =========================
# 11. Збереження результатів
# =========================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    eval_df.to_excel(writer, sheet_name="Evaluated Data", index=False)
    conf_matrix_with_total.to_excel(writer, sheet_name="Confusion Matrix")
    (conf_matrix_pct_total_with_total * 100).round(2).to_excel(
        writer,
        sheet_name="Confusion Matrix Percent"
    )
    class_metrics_df.to_excel(writer, sheet_name="Class Accuracy", index=False)
    summary_metrics.to_excel(writer, sheet_name="Summary Metrics", index=False)
    slide_metrics.to_excel(writer, sheet_name="Slide Metrics", index=False)

print(f"\nФайл збережено: {OUTPUT_FILE}")