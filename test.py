import pandas as pd

patterns = good.copy().reset_index(drop=True)

COV_EPS = 0.01  # допустима різниця coverage

# сортуємо: довші + з більшим coverage — першими
patterns = patterns.sort_values(
    by=["acq_coverage", "len"],
    ascending=[False, False]
).reset_index(drop=True)

keep = [True] * len(patterns)

for i in range(len(patterns)):
    if not keep[i]:
        continue

    pi = patterns.loc[i, "pattern"]
    ci = patterns.loc[i, "acq_coverage"]

    for j in range(i + 1, len(patterns)):
        if not keep[j]:
            continue

        pj = patterns.loc[j, "pattern"]
        cj = patterns.loc[j, "acq_coverage"]

        if abs(ci - cj) < COV_EPS and pj in pi:
            keep[j] = False

patterns_merged = patterns[keep].reset_index(drop=True)
patterns_merged.head(20)
