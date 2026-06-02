thresholds = {
    "p_liabs": 0.30,
    "p_assets": 0.30,
    "p_fx": 0.59
}

def scale_by_threshold(p, t):
    if pd.isna(p):
        return 0
    
    if p < t:
        return 0.5 * p / t
    else:
        return 0.5 + 0.5 * (p - t) / (1 - t)

rec["score_liabs"] = rec["p_liabs"].apply(lambda x: scale_by_threshold(x, thresholds["p_liabs"]))
rec["score_assets"] = rec["p_assets"].apply(lambda x: scale_by_threshold(x, thresholds["p_assets"]))
rec["score_fx"] = rec["p_fx"].apply(lambda x: scale_by_threshold(x, thresholds["p_fx"]))



score_cols = ["score_liabs", "score_assets", "score_fx"]

rec["recommendation_score"] = rec[score_cols].max(axis=1)

rec["recommended_product"] = rec[score_cols].idxmax(axis=1).map({
    "score_liabs": "Liabilities",
    "score_assets": "Assets",
    "score_fx": "FX"
})



rec["recommended_raw_probability"] = rec.apply(
    lambda row: {
        "Liabilities": row["p_liabs"],
        "Assets": row["p_assets"],
        "FX": row["p_fx"]
    }[row["recommended_product"]],
    axis=1
)


def explain_scaled_recommendation(row):
    product = row["recommended_product"]

    if product == "Liabilities":
        return (
            f"Рекомендовано Liabilities, бо нормалізований score = {row['score_liabs']:.2f}. "
            f"Сира ймовірність = {row['p_liabs']:.2f}, оптимальний threshold моделі = 0.30."
        )

    if product == "Assets":
        return (
            f"Рекомендовано Assets, бо нормалізований score = {row['score_assets']:.2f}. "
            f"Сира ймовірність = {row['p_assets']:.2f}, оптимальний threshold моделі = 0.30."
        )

    if product == "FX":
        return (
            f"Рекомендовано FX, бо нормалізований score = {row['score_fx']:.2f}. "
            f"Сира ймовірність = {row['p_fx']:.2f}, оптимальний threshold моделі = 0.59."
        )

rec["explanation"] = rec.apply(explain_scaled_recommendation, axis=1)


