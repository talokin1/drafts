import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.float_format", "{:,.4f}".format)


def target_basic_stats(y: pd.Series, name: str):
    y = y.astype(float)

    stats = {
        "n": len(y),
        "zeros_share": (y == 0).mean(),
        "negatives_cnt": (y < 0).sum(),
        "min": y.min(),
        "p01": y.quantile(0.01),
        "p05": y.quantile(0.05),
        "median": y.median(),
        "p90": y.quantile(0.90),
        "p95": y.quantile(0.95),
        "p99": y.quantile(0.99),
        "p995": y.quantile(0.995),
        "max": y.max(),
    }

    print(f"\n=== {name} BASIC STATS ===")
    for k, v in stats.items():
        print(f"{k:15}: {v}")

    return stats


def target_eda(y: pd.Series, name: str):
    y = y.astype(float)

    y_pos = y[y > 0]
    y_zero = y[y == 0]

    print(f"\n=== {name} ===")
    print(f"Total: {len(y)}")
    print(f"Zeros: {len(y_zero)} ({len(y_zero)/len(y):.2%})")
    print(f"Positive: {len(y_pos)} ({len(y_pos)/len(y):.2%})")
    print(f"Negatives: {(y < 0).sum()}")

    # ---------- HISTOGRAMS ----------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Full distribution (linear)
    axes[0, 0].hist(y, bins=200)
    axes[0, 0].set_title(f"{name} – full (linear)")

    # 2. Positive only (linear)
    axes[0, 1].hist(y_pos, bins=200)
    axes[0, 1].set_title(f"{name} – positive only (linear)")

    # 3. Positive only (log y)
    axes[0, 2].hist(np.log1p(y_pos), bins=200)
    axes[0, 2].set_title(f"{name} – log1p(y), positive only")

    # ---------- ECDF ----------
    y_sorted = np.sort(y_pos)
    ecdf = np.arange(1, len(y_sorted)+1) / len(y_sorted)

    axes[1, 0].plot(y_sorted, ecdf)
    axes[1, 0].set_title(f"{name} – ECDF (positive)")
    axes[1, 0].set_xscale("log")

    # ---------- BOXPLOTS ----------
    sns.boxplot(x=y_pos, ax=axes[1, 1])
    axes[1, 1].set_title(f"{name} – boxplot (positive)")

    sns.boxplot(x=np.log1p(y_pos), ax=axes[1, 2])
    axes[1, 2].set_title(f"{name} – boxplot log1p(y)")

    plt.tight_layout()
    plt.show()



def zero_positive_split(df: pd.DataFrame, target: str, by: list[str]):
    tmp = df.copy()
    tmp["is_positive"] = (tmp[target] > 0).astype(int)

    for col in by:
        agg = (
            tmp
            .groupby(col)["is_positive"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )

        print(f"\n=== Zero vs Positive by {col} ===")
        display(agg.head(20))


def concentration_analysis(y: pd.Series, name: str):
    y_pos = y[y > 0].sort_values(ascending=False)

    total = y_pos.sum()

    for q in [0.001, 0.005, 0.01, 0.05]:
        top_k = int(len(y_pos) * q)
        share = y_pos.iloc[:top_k].sum() / total if top_k > 0 else 0
        print(f"{name}: top {q*100:.2f}% → {share:.2%} of total sum")





# CURR_ACC
target_basic_stats(df["CURR_ACC"], "CURR_ACC")
target_eda(df["CURR_ACC"], "CURR_ACC")
concentration_analysis(df["CURR_ACC"], "CURR_ACC")

# TERM_DEPOSITS
target_basic_stats(df["TERM_DEPOSITS"], "TERM_DEPOSITS")
target_eda(df["TERM_DEPOSITS"], "TERM_DEPOSITS")
concentration_analysis(df["TERM_DEPOSITS"], "TERM_DEPOSITS")
