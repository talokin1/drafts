DEFAULT_THRESHOLDS = {
    "LIABILITIES": 0.30,
    "ASSETS": 0.30,
    "FX": 0.59
}

initial_thresholds = {}

for product, config in PRODUCTS.items():
    target_col = config["target"]

    positives = validation_tune[target_col].fillna(0).astype(int).sum()

    if positives == 0:
        print(
            f"{product}: у tune немає позитивних {target_col}, "
            f"використовуємо default threshold"
        )
        initial_thresholds[product] = DEFAULT_THRESHOLDS[product]
    else:
        initial_thresholds[product] = find_pr_threshold(
            validation_tune,
            config["score"],
            target_col
        )