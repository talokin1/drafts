import pandas as pd
import numpy as np


rec_product_col = 'recomended_product'

print(f"Колонка з рекомендацією: {rec_product_col}")


# =========================================================
# 3. Формуємо фактичні продукти, до яких залучився клієнт
# =========================================================

validation_all["NEW_LIABILITIES"] = (
    validation_all["NEW_LIABILITIES"]
    .fillna(False)
    .astype(bool)
)

validation_all["NEW_ASSETS"] = (
    validation_all["NEW_ASSETS"]
    .fillna(False)
    .astype(bool)
)

validation_all["NEW_FX"] = (
    validation_all["NEW_FX"]
    .fillna(False)
    .astype(bool)
)

validation_all["actual_products_list"] = validation_all.apply(
    lambda row: [
        product
        for product, condition in {
            "Liabilities": row["NEW_LIABILITIES"],
            "Assets": row["NEW_ASSETS"],
            "FX": row["NEW_FX"]
        }.items()
        if condition
    ],
    axis=1
)

validation_all["n_actual_products"] = validation_all["actual_products_list"].apply(len)

validation_all["actual_product"] = validation_all["actual_products_list"].apply(
    lambda x: ", ".join(x) if len(x) > 0 else pd.NA
)

validation_all["was_engaged"] = validation_all["n_actual_products"] > 0


validation_engaged = validation_all.loc[
    validation_all["was_engaged"]
].copy()

print(f"Усього рядків у validation_all: {len(validation_all):,}")
print(f"Клієнтів / рядків із фактичним залученням: {len(validation_engaged):,}")

display(
    validation_engaged[
        [
            "IDENTIFYCODE",
            "NEW_LIABILITIES",
            "NEW_ASSETS",
            "NEW_FX",
            "n_actual_products",
            "actual_product"
        ]
    ].head(10)
)