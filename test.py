import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =========================
# 1. COPY DATA
# =========================
df = data.copy()

# =========================
# 2. TARGET
# =========================
TARGET = "MONTHLY_INCOME"

df = df[df[TARGET].notna()]
y = np.log1p(df[TARGET])
X = df.drop(columns=[TARGET])

# =========================
# 3. DROP TECHNICAL IDS
# =========================
drop_cols = [
    "CONTRAGENTID",
    "IDENTIFYCODE"
]

X = X.drop(columns=[c for c in drop_cols if c in X.columns])

# =========================
# 4. COLUMN GROUPS
# =========================
categorical_cols = [
    "SEGMENT",
    "FIRM_KVED"
]

ratio_cols = [
    "PRIMARY_LIABS",
    "0_LIABS",
    "0-100K_LIABS",
    "100K-500K_LIABS",
    "5M-10M_LIABS",
    ">10M_LIABS",
    "10M+_LIABS",
    "RATIO_OTP_OTH_2025_11"
]

monetary_cols = [
    "OTP_AV_LIAB_2025_11",
    "OTP_TURN_2025_11",
    "OTH_BANKS_TURN_2025_11",
    "LOANS_AVG",
    "LOANS_LIMIT",
    "ASSETS_SUIT_AMT"
]

count_cols = [
    "NB_CARDS_2025_11",
    "NB_ACCOUNTS_2025_11",
    "NB_SAVINGS_2025_11",
    "NB_DEPOSITS_2025_11",
    "NB_ACCOUNTS_OVER_2",
    "NB_LOANS_2025_11"
]

score_cols = [
    "MSB_SCORE"
]

# =========================
# 5. TRANSFORMERS
# =========================
log_transformer = FunctionTransformer(
    func=lambda x: np.log1p(x),
    feature_names_out="one-to-one"
)

cat_transformer = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

# =========================
# 6. COLUMN TRANSFORMER
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, categorical_cols),
        ("log_money", log_transformer, monetary_cols),
        ("ratio", "passthrough", ratio_cols),
        ("counts", "passthrough", count_cols),
        ("scores", "passthrough", score_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# =========================
# 7. PIPELINE (READY FOR LGB)
# =========================
pipeline = Pipeline(
    steps=[
        ("prep", preprocessor)
    ]
)

# =========================
# 8. TRAIN / VALID SPLIT
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train_prep = pipeline.fit_transform(X_train)
X_valid_prep = pipeline.transform(X_valid)

feature_names = pipeline.named_steps["prep"].get_feature_names_out()
