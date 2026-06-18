import numpy as np
import pandas as pd

ID_COL = "IDENTIFYCODE"

# ============================================================
# 1. CONFIG
# ============================================================

LIABS_FEATURES = [
    "CASH_CUR",
    "CASH_PREV",
    "CASH_DIF",
    "ALR%_CUR",
    "ALR%_PREV",
    "ALR%_DIF",
    "ASSETS_PREV",
    "ASSETS_DIF",
    "ROE%_CUR",
    "ROE%_DIF",
    "TAT_CUR",
    "NB_EMP",
    "CDR_PREV"
]

ASSETS_FEATURES = [
    "CURRENT_ASSETS_CUR",
    "ASSETS_CUR",
    "INVENTORY_CUR",
    "PAYABLES_DIF",
    "DSO_PREV",
    "CR%_DIF",
    "LIQUID_ASSETS_DIF",
    "OPM%_CUR",
    "OPM%_DIF",
    "FIXED_ASSETS_PREV",
    "FIXED_ASSETS_DIF",
    "LTFR_DIF"
]

FX_FEATURES = [
    "IMPORT_USD",
    "EXPORT_USD",
    "A1165",
    "A2120",
    "A2160",
    "A2285",
    "DIO_CUR",
    "DIO_PREV",
    "TAT_CUR",
    "B2120"
]


# Ваги між propensity-моделлю та фінансовими індикаторами
PRODUCT_WEIGHTS = {
    "Liabilities": {"model": 0.70, "fin": 0.30},
    "Assets": {"model": 0.65, "fin": 0.35},
    "FX": {"model": 0.80, "fin": 0.20},
}


# Напрям показника:
# high -> більше значення краще
# low  -> менше значення краще
# abs  -> важливий сам масштаб відхилення, знак неважливий
FEATURE_DIRECTIONS = {
    # Liabilities
    "CASH_CUR": "high",
    "CASH_PREV": "high",
    "CASH_DIF": "high",
    "ALR%_CUR": "high",
    "ALR%_PREV": "high",
    "ALR%_DIF": "high",
    "ASSETS_PREV": "high",
    "ASSETS_DIF": "high",
    "ROE%_CUR": "high",
    "ROE%_DIF": "high",
    "TAT_CUR": "high",
    "NB_EMP": "high",
    "CDR_PREV": "high",

    # Assets
    "CURRENT_ASSETS_CUR": "high",
    "ASSETS_CUR": "high",
    "INVENTORY_CUR": "high",
    "PAYABLES_DIF": "high",
    "DSO_PREV": "high",
    "CR%_DIF": "high",
    "LIQUID_ASSETS_DIF": "high",
    "OPM%_CUR": "high",
    "OPM%_DIF": "high",
    "FIXED_ASSETS_PREV": "high",
    "FIXED_ASSETS_DIF": "high",
    "LTFR_DIF": "high",

    # FX
    "IMPORT_USD": "high",
    "EXPORT_USD": "high",
    "A1165": "high",
    "A2120": "high",
    "A2160": "high",
    "A2285": "high",
    "DIO_CUR": "high",
    "DIO_PREV": "high",
    "B2120": "high",
}


# Ваги фічей всередині продуктового financial score
# Чим більш прямий бізнес-сигнал, тим більша вага
LIABS_FEATURE_WEIGHTS = {
    "CASH_CUR": 0.18,
    "CASH_PREV": 0.10,
    "CASH_DIF": 0.12,
    "ALR%_CUR": 0.12,
    "ALR%_PREV": 0.06,
    "ALR%_DIF": 0.08,
    "ASSETS_PREV": 0.08,
    "ASSETS_DIF": 0.08,
    "ROE%_CUR": 0.06,
    "ROE%_DIF": 0.04,
    "TAT_CUR": 0.04,
    "NB_EMP": 0.02,
    "CDR_PREV": 0.02,
}

ASSETS_FEATURE_WEIGHTS = {
    "CURRENT_ASSETS_CUR": 0.14,
    "ASSETS_CUR": 0.08,
    "INVENTORY_CUR": 0.12,
    "PAYABLES_DIF": 0.10,
    "DSO_PREV": 0.10,
    "CR%_DIF": 0.08,
    "LIQUID_ASSETS_DIF": 0.08,
    "OPM%_CUR": 0.10,
    "OPM%_DIF": 0.08,
    "FIXED_ASSETS_PREV": 0.08,
    "FIXED_ASSETS_DIF": 0.08,
    "LTFR_DIF": 0.06,
}

FX_FEATURE_WEIGHTS = {
    "IMPORT_USD": 0.28,
    "EXPORT_USD": 0.28,
    "A1165": 0.08,
    "A2120": 0.08,
    "A2160": 0.08,
    "A2285": 0.08,
    "DIO_CUR": 0.04,
    "DIO_PREV": 0.03,
    "TAT_CUR": 0.03,
    "B2120": 0.02,
}

def normalize_identifycode(s):
    return (
        s.astype(str)
         .str.replace(r"\.0$", "", regex=True)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str.zfill(8)
    )


def prepare_financial_data(fin_ind, fin_ind_wo=None):
    fin_1 = fin_ind.copy()
    fin_1[ID_COL] = normalize_identifycode(fin_1[ID_COL])

    if fin_ind_wo is not None:
        fin_2 = fin_ind_wo.copy()
        fin_2[ID_COL] = normalize_identifycode(fin_2[ID_COL])
        fin = pd.concat([fin_1, fin_2], ignore_index=True, sort=False)
    else:
        fin = fin_1.copy()

    selected_features = sorted(
        set(LIABS_FEATURES + ASSETS_FEATURES + FX_FEATURES)
    )

    existing_features = [c for c in selected_features if c in fin.columns]

    print("Existing financial features:", len(existing_features))
    print(existing_features)

    missing_features = [c for c in selected_features if c not in fin.columns]

    if missing_features:
        print("\nMissing financial features:")
        print(missing_features)

    # Обираємо найбільш заповнений запис по компанії
    fin["_non_null_cnt"] = fin[existing_features].notna().sum(axis=1)

    sort_cols = [ID_COL, "_non_null_cnt"]
    ascending = [True, True]

    # Якщо є рік/дата звітності — беремо найсвіжіший з більш заповнених
    if "REP_YEAR" in fin.columns:
        sort_cols.append("REP_YEAR")
        ascending.append(True)

    if "DFILL" in fin.columns:
        sort_cols.append("DFILL")
        ascending.append(True)

    fin = (
        fin.sort_values(sort_cols, ascending=ascending)
           .drop_duplicates(ID_COL, keep="last")
           .copy()
    )

    keep_cols = [ID_COL] + existing_features

    return fin[keep_cols].copy(), existing_features


fin_scoring, existing_fin_features = prepare_financial_data(fin_ind, fin_ind_wo)








def percentile_score(df, col, direction="high"):
    x = pd.to_numeric(df[col], errors="coerce")

    if direction == "abs":
        x = x.abs()

    valid = x.dropna()

    if valid.empty:
        return pd.Series(np.nan, index=df.index)

    # winsorization: обрізаємо екстремальні хвости
    q01 = valid.quantile(0.01)
    q99 = valid.quantile(0.99)

    x = x.clip(q01, q99)

    if direction == "low":
        x = -x

    return x.rank(pct=True)


def add_feature_scores(fin_df, features):
    fin = fin_df.copy()

    for col in features:
        if col not in fin.columns:
            continue

        direction = FEATURE_DIRECTIONS.get(col, "high")
        score_col = f"{col}_score"

        fin[score_col] = percentile_score(fin, col, direction=direction)

    return fin


fin_scoring = add_feature_scores(fin_scoring, existing_fin_features)







def product_fin_score(fin_df, features, feature_weights, out_col):
    fin = fin_df.copy()

    weighted_sum = pd.Series(0.0, index=fin.index)
    weight_sum = pd.Series(0.0, index=fin.index)

    used_features = []

    for feature in features:
        score_col = f"{feature}_score"

        if score_col not in fin.columns:
            continue

        w = feature_weights.get(feature, 1.0)

        mask = fin[score_col].notna()

        weighted_sum.loc[mask] += fin.loc[mask, score_col] * w
        weight_sum.loc[mask] += w

        used_features.append(feature)

    fin[out_col] = weighted_sum / weight_sum.replace(0, np.nan)

    print(f"{out_col}: used {len(used_features)} features")
    print(used_features)

    return fin, used_features


fin_scoring, used_liabs_features = product_fin_score(
    fin_scoring,
    LIABS_FEATURES,
    LIABS_FEATURE_WEIGHTS,
    "score_liabs_fin"
)

fin_scoring, used_assets_features = product_fin_score(
    fin_scoring,
    ASSETS_FEATURES,
    ASSETS_FEATURE_WEIGHTS,
    "score_assets_fin"
)

fin_scoring, used_fx_features = product_fin_score(
    fin_scoring,
    FX_FEATURES,
    FX_FEATURE_WEIGHTS,
    "score_fx_fin"
)

# На випадок, якщо у rec IDENTIFYCODE ще не нормалізований
rec[ID_COL] = normalize_identifycode(rec[ID_COL])

fin_cols_for_merge = [
    ID_COL,
    "score_liabs_fin",
    "score_assets_fin",
    "score_fx_fin"
]

# Додаємо також component scores для пояснення рекомендацій
component_score_cols = [
    c for c in fin_scoring.columns
    if c.endswith("_score") and c not in fin_cols_for_merge
]

fin_cols_for_merge += component_score_cols

rec = rec.merge(
    fin_scoring[fin_cols_for_merge],
    on=ID_COL,
    how="left"
)


# Якщо у тебе вже є score_liabs / score_assets / score_fx:
if "score_liabs_model" not in rec.columns:
    rec["score_liabs_model"] = rec["score_liabs"]

if "score_assets_model" not in rec.columns:
    rec["score_assets_model"] = rec["score_assets"]

if "score_fx_model" not in rec.columns:
    rec["score_fx_model"] = rec["score_fx"]


def combine_scores(model_score, fin_score, model_weight, fin_weight):
    model_score = pd.to_numeric(model_score, errors="coerce").fillna(0)
    fin_score = pd.to_numeric(fin_score, errors="coerce")

    # Якщо фінансових даних немає — не штрафуємо клієнта, залишаємо тільки model score
    return np.where(
        fin_score.notna(),
        model_weight * model_score + fin_weight * fin_score,
        model_score
    )


rec["final_score_liabs"] = combine_scores(
    rec["score_liabs_model"],
    rec["score_liabs_fin"],
    PRODUCT_WEIGHTS["Liabilities"]["model"],
    PRODUCT_WEIGHTS["Liabilities"]["fin"]
)

rec["final_score_assets"] = combine_scores(
    rec["score_assets_model"],
    rec["score_assets_fin"],
    PRODUCT_WEIGHTS["Assets"]["model"],
    PRODUCT_WEIGHTS["Assets"]["fin"]
)

rec["final_score_fx"] = combine_scores(
    rec["score_fx_model"],
    rec["score_fx_fin"],
    PRODUCT_WEIGHTS["FX"]["model"],
    PRODUCT_WEIGHTS["FX"]["fin"]
)



PRODUCTS = np.array(["Liabilities", "Assets", "FX"])

FINAL_SCORE_COLS = [
    "final_score_liabs",
    "final_score_assets",
    "final_score_fx"
]

score_matrix = rec[FINAL_SCORE_COLS].to_numpy()

order = np.argsort(-score_matrix, axis=1)
sorted_scores = np.take_along_axis(score_matrix, order, axis=1)

rec["recommended_product"] = PRODUCTS[order[:, 0]]
rec["alternative_product"] = PRODUCTS[order[:, 1]]
rec["third_product"] = PRODUCTS[order[:, 2]]

rec["recommendation_score"] = sorted_scores[:, 0]
rec["alternative_score"] = sorted_scores[:, 1]
rec["third_score"] = sorted_scores[:, 2]

rec["score_gap"] = rec["recommendation_score"] - rec["alternative_score"]

rec["recommendation_strength"] = np.select(
    [
        rec["recommendation_score"] < 0.50,
        rec["score_gap"] < 0.05,
        rec["score_gap"] >= 0.15
    ],
    [
        "no_strong_recommendation",
        "ambiguous",
        "strong"
    ],
    default="medium"
)

rec["final_recommendation"] = np.where(
    rec["recommendation_score"] < 0.50,
    "No strong recommendation",
    rec["recommended_product"]
)















MODEL_SCORE_COLS = [
    "score_liabs_model",
    "score_assets_model",
    "score_fx_model"
]

model_matrix = rec[MODEL_SCORE_COLS].to_numpy()
model_order = np.argsort(-model_matrix, axis=1)

rec["model_only_product"] = PRODUCTS[model_order[:, 0]]
rec["changed_by_financial_indicators"] = (
    rec["model_only_product"] != rec["recommended_product"]
)

diagnostics = pd.DataFrame({
    "metric": [
        "n_clients",
        "share_changed_by_fin_indicators",
        "avg_score_gap",
        "share_no_strong_recommendation",
        "share_ambiguous",
        "share_strong"
    ],
    "value": [
        len(rec),
        rec["changed_by_financial_indicators"].mean(),
        rec["score_gap"].mean(),
        (rec["recommendation_strength"] == "no_strong_recommendation").mean(),
        (rec["recommendation_strength"] == "ambiguous").mean(),
        (rec["recommendation_strength"] == "strong").mean()
    ]
})

display(diagnostics)

display(
    rec["final_recommendation"]
    .value_counts(normalize=True)
    .rename_axis("product")
    .reset_index(name="share")
)

PRODUCT_FEATURES = {
    "Liabilities": used_liabs_features,
    "Assets": used_assets_features,
    "FX": used_fx_features
}

PRODUCT_FIN_SCORE_COL = {
    "Liabilities": "score_liabs_fin",
    "Assets": "score_assets_fin",
    "FX": "score_fx_fin"
}

PRODUCT_MODEL_SCORE_COL = {
    "Liabilities": "score_liabs_model",
    "Assets": "score_assets_model",
    "FX": "score_fx_model"
}

PRODUCT_FINAL_SCORE_COL = {
    "Liabilities": "final_score_liabs",
    "Assets": "final_score_assets",
    "FX": "final_score_fx"
}


def get_top_fin_drivers(row, product, n=3):
    features = PRODUCT_FEATURES.get(product, [])

    vals = []

    for feature in features:
        score_col = f"{feature}_score"

        if score_col in row.index and pd.notna(row[score_col]):
            vals.append((feature, row[score_col]))

    if not vals:
        return "фінансові індикатори відсутні"

    vals = sorted(vals, key=lambda x: x[1], reverse=True)[:n]

    return "; ".join([f"{feature}: {score:.2f}" for feature, score in vals])


def explain_recommendation(row):
    product = row["recommended_product"]

    model_score_col = PRODUCT_MODEL_SCORE_COL[product]
    fin_score_col = PRODUCT_FIN_SCORE_COL[product]
    final_score_col = PRODUCT_FINAL_SCORE_COL[product]

    model_score = row[model_score_col]
    fin_score = row[fin_score_col]
    final_score = row[final_score_col]

    fin_score_text = f"{fin_score:.2f}" if pd.notna(fin_score) else "немає даних"

    drivers = get_top_fin_drivers(row, product)

    return (
        f"Рекомендовано {product}. "
        f"Model score = {model_score:.2f}; "
        f"financial score = {fin_score_text}; "
        f"final score = {final_score:.2f}. "
        f"Основні фінансові драйвери: {drivers}. "
        f"Альтернатива: {row['alternative_product']} "
        f"(score = {row['alternative_score']:.2f})."
    )


rec["explanation"] = rec.apply(explain_recommendation, axis=1)










final_cols = [
    ID_COL,

    "p_liabs",
    "p_assets",
    "p_fx",

    "score_liabs_model",
    "score_assets_model",
    "score_fx_model",

    "score_liabs_fin",
    "score_assets_fin",
    "score_fx_fin",

    "final_score_liabs",
    "final_score_assets",
    "final_score_fx",

    "recommended_product",
    "alternative_product",
    "recommendation_score",
    "alternative_score",
    "score_gap",
    "recommendation_strength",
    "final_recommendation",

    "model_only_product",
    "changed_by_financial_indicators",

    "explanation"
]

final_cols = [c for c in final_cols if c in rec.columns]

rec_final = rec[final_cols].copy()

rec_final.head(20)