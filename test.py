# назви колонок
TARGET_COL = y_train.name if hasattr(y_train, "name") else "TARGET"
PRED_COL = "MODEL_PRED"

# train
x_train_total = x_train_total.rename(columns={x_train_total.columns[-2]: TARGET_COL})

# valid
x_valid_total = x_valid_total.rename(columns={x_valid_total.columns[-2]: TARGET_COL})


x_train_total.to_csv(
    "regression_train_results.csv",
    index=False,
    encoding="utf-8"
)

x_valid_total.to_csv(
    "regression_valid_results.csv",
    index=False,
    encoding="utf-8"
)


BINS = [0, 5_000, 10_000, 20_000, 30_000, 50_000, np.inf]
BIN_LABELS = [
    "0-5k",
    "5-10k",
    "10-20k",
    "20-30k",
    "30-50k",
    "50k+"
]


x_train_total["INCOME_BIN"] = pd.cut(
    x_train_total[TARGET_COL],
    bins=BINS,
    labels=BIN_LABELS,
    right=False
)

x_valid_total["INCOME_BIN"] = pd.cut(
    x_valid_total[TARGET_COL],
    bins=BINS,
    labels=BIN_LABELS,
    right=False
)


x_train_total["PRED_BIN"] = pd.cut(
    x_train_total[PRED_COL],
    bins=BINS,
    labels=BIN_LABELS,
    right=False
)

x_valid_total["PRED_BIN"] = pd.cut(
    x_valid_total[PRED_COL],
    bins=BINS,
    labels=BIN_LABELS,
    right=False
)


pd.crosstab(
    x_valid_total["INCOME_BIN"],
    x_valid_total["PRED_BIN"],
    normalize="index"
)







x_train_total.to_csv(
    "regression_train_with_bins.csv",
    index=False,
    encoding="utf-8"
)

x_valid_total.to_csv(
    "regression_valid_with_bins.csv",
    index=False,
    encoding="utf-8"
)
