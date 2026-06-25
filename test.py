# =========================================================
# Налаштування: допустима різниця між 1-м та 2-м продуктом
# =========================================================

MULTI_PRODUCT_GAP = 0.05
# 0.05 = рекомендувати два продукти,
# якщо їх нормалізовані скори відрізняються не більше ніж на 5 п.п.

score_cols = ["score_liabs", "score_assets", "score_fx"]

score_to_product = {
    "score_liabs": "Liabilities",
    "score_assets": "Assets",
    "score_fx": "FX"
}


# =========================================================
# Формування top-1 / top-2 рекомендації
# =========================================================

def get_recommended_products(row, gap=MULTI_PRODUCT_GAP):
    
    scores = row[score_cols].copy()
    
    # На випадок пропусків
    scores = scores.fillna(0)
    
    # Сортуємо скори від найбільшого до найменшого
    scores_sorted = scores.sort_values(ascending=False)
    
    best_score = scores_sorted.iloc[0]
    second_score = scores_sorted.iloc[1]
    
    # Перший продукт завжди додаємо
    selected_score_cols = [scores_sorted.index[0]]
    
    # Якщо другий продукт достатньо близький до першого — додаємо його
    if (best_score - second_score) <= gap:
        selected_score_cols.append(scores_sorted.index[1])
    
    # Перетворюємо назви скорів у людські назви продуктів
    selected_products = [
        score_to_product[col]
        for col in selected_score_cols
    ]
    
    return ", ".join(selected_products)


# Найкращий нормалізований скор
rec["recommendation_score"] = rec[score_cols].max(axis=1)

# Рекомендація: один або два продукти
rec["recommended_product"] = rec.apply(
    get_recommended_products,
    axis=1
)

# Для аналітики: скільки продуктів потрапило в рекомендацію
rec["n_recommended_products"] = rec["recommended_product"].str.count(",") + 1