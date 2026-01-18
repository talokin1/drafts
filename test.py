def target_sanity(y: pd.Series, name: str):
    print(f"=== {name} ===")
    print("n:", len(y))
    print("zeros:", (y == 0).mean())
    print("negatives:", (y < 0).sum())
    print("min:", y.min())
    print("median:", y.median())
    print("p90:", y.quantile(0.90))
    print("p99:", y.quantile(0.99))
    print("max:", y.max())


target_sanity(targets["CURR_ACC"], "CURR_ACC")
target_sanity(targets["TERM_DEPOSITS"], "TERM_DEPOSITS")
