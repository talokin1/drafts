candidates = []

for product in PRODUCT_NAMES:
    score = row[f"{product}_PCT"]
    threshold = product_thresholds[product]

    if score >= threshold:
        margin = (score - threshold) / max(1 - threshold, 1e-6)
        candidates.append((product, margin))

if not candidates:
    predictions.append({"NOTHING_TO_DO"})
    continue

candidates.sort(key=lambda x: x[1], reverse=True)

result = {candidates[0][0]}

if (
    len(candidates) > 1
    and candidates[0][1] - candidates[1][1] <= tie_delta
):
    result.add(candidates[1][0])

predictions.append(result)