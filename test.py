def get_recommended_raw_probabilities(row):
    
    # Розбиваємо "Liabilities, Assets" -> ["Liabilities", "Assets"]
    products = [
        product.strip()
        for product in str(row["recommended_product"]).split(",")
    ]
    
    product_to_probability = {
        "Liabilities": row["p_liabs"],
        "Assets": row["p_assets"],
        "FX": row["p_fx"]
    }
    
    # Повертаємо ймовірності тільки рекомендованих продуктів
    result = [
        f"{product}: {product_to_probability[product]:.3f}"
        for product in products
        if product in product_to_probability
    ]
    
    return ", ".join(result)


rec["recommended_raw_probability"] = rec.apply(
    get_recommended_raw_probabilities,
    axis=1
)






def explain_scaled_recommendation(row):
    
    products = [
        product.strip()
        for product in str(row["recommended_product"]).split(",")
    ]
    
    product_info = {
        "Liabilities": {
            "score": row["score_liabs"],
            "probability": row["p_liabs"],
            "threshold": thresholds["p_liabs"]
        },
        "Assets": {
            "score": row["score_assets"],
            "probability": row["p_assets"],
            "threshold": thresholds["p_assets"]
        },
        "FX": {
            "score": row["score_fx"],
            "probability": row["p_fx"],
            "threshold": thresholds["p_fx"]
        }
    }
    
    explanations = []
    
    for product in products:
        
        info = product_info[product]
        
        explanations.append(
            f"{product}: "
            f"нормалізований score = {info['score']:.2f}, "
            f"сира ймовірність = {info['probability']:.2f}, "
            f"threshold = {info['threshold']:.2f}"
        )
    
    return " | ".join(explanations)


rec["explanation"] = rec.apply(
    explain_scaled_recommendation,
    axis=1
)