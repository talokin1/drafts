import numpy as np
import pandas as pd
import joblib
import os


class TwoStageFXModel:
    def __init__(
        self,
        clf_model,
        reg_model,
        clf_features,
        reg_features,
        clf_cat_cols,
        reg_cat_cols,
        clf_num_medians,
        reg_num_medians,
        clf_category_values,
        reg_category_values,
        threshold=0.30,
        prediction_mode="expected"  # "expected", "threshold", "soft_expected"
    ):
        self.clf_model = clf_model
        self.reg_model = reg_model

        self.clf_features = list(clf_features)
        self.reg_features = list(reg_features)

        self.clf_cat_cols = list(clf_cat_cols)
        self.reg_cat_cols = list(reg_cat_cols)

        self.clf_num_cols = [c for c in self.clf_features if c not in self.clf_cat_cols]
        self.reg_num_cols = [c for c in self.reg_features if c not in self.reg_cat_cols]

        self.clf_num_medians = clf_num_medians
        self.reg_num_medians = reg_num_medians

        self.clf_category_values = clf_category_values
        self.reg_category_values = reg_category_values

        self.threshold = threshold
        self.prediction_mode = prediction_mode

    def _prepare_X(
        self,
        df,
        features,
        cat_cols,
        num_cols,
        num_medians,
        category_values,
        model_name="model"
    ):
        X = df.copy()

        missing_cols = [c for c in features if c not in X.columns]
        if len(missing_cols) > 0:
            raise ValueError(
                f"У df немає потрібних фіч для {model_name}: {missing_cols}"
            )

        X = X[features].copy()

        # numeric
        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

            fill_value = num_medians.get(c, 0)
            if pd.isna(fill_value):
                fill_value = 0

            X[c] = X[c].fillna(fill_value)

        # categorical
        for c in cat_cols:
            X[c] = X[c].astype("string").fillna("UNKNOWN")

            cats = list(category_values.get(c, ["UNKNOWN"]))

            if "UNKNOWN" not in cats:
                cats.append("UNKNOWN")

            X[c] = X[c].where(X[c].isin(cats), "UNKNOWN")
            X[c] = pd.Categorical(X[c], categories=cats)

        return X

    def _prepare_clf_X(self, df):
        return self._prepare_X(
            df=df,
            features=self.clf_features,
            cat_cols=self.clf_cat_cols,
            num_cols=self.clf_num_cols,
            num_medians=self.clf_num_medians,
            category_values=self.clf_category_values,
            model_name="classifier"
        )

    def _prepare_reg_X(self, df):
        return self._prepare_X(
            df=df,
            features=self.reg_features,
            cat_cols=self.reg_cat_cols,
            num_cols=self.reg_num_cols,
            num_medians=self.reg_num_medians,
            category_values=self.reg_category_values,
            model_name="regressor"
        )

    def predict(self, df):
        X_clf = self._prepare_clf_X(df)
        X_reg = self._prepare_reg_X(df)

        # 1. Probability of FX activity
        proba = self.clf_model.predict_proba(X_clf)[:, 1]

        # 2. Conditional FX amount
        pred_log = self.reg_model.predict(X_reg)
        pred_log = np.clip(pred_log, 0, None)

        fx_cond_pred = np.expm1(pred_log)

        # 3. Final potential
        fx_expected = proba * fx_cond_pred

        fx_threshold = np.zeros(len(df))
        mask = proba >= self.threshold
        fx_threshold[mask] = fx_cond_pred[mask]

        fx_soft_expected = fx_cond_pred * (0.3 + 0.7 * proba)

        if self.prediction_mode == "expected":
            fx_potential = fx_expected
        elif self.prediction_mode == "threshold":
            fx_potential = fx_threshold
        elif self.prediction_mode == "soft_expected":
            fx_potential = fx_soft_expected
        else:
            raise ValueError("prediction_mode має бути 'expected', 'threshold' або 'soft_expected'")

        return fx_potential, proba

    def predict_full(self, df):
        X_clf = self._prepare_clf_X(df)
        X_reg = self._prepare_reg_X(df)

        proba = self.clf_model.predict_proba(X_clf)[:, 1]

        pred_log = self.reg_model.predict(X_reg)
        pred_log = np.clip(pred_log, 0, None)

        fx_cond_pred = np.expm1(pred_log)

        fx_expected = proba * fx_cond_pred

        fx_threshold = np.zeros(len(df))
        mask = proba >= self.threshold
        fx_threshold[mask] = fx_cond_pred[mask]

        fx_soft_expected = fx_cond_pred * (0.3 + 0.7 * proba)

        result = df.copy()

        result["PROB_TO_FX"] = proba
        result["FX_COND_PRED"] = fx_cond_pred
        result["FX_EXPECTED"] = fx_expected
        result["FX_THRESHOLD_PRED"] = fx_threshold
        result["FX_SOFT_EXPECTED"] = fx_soft_expected

        if self.prediction_mode == "expected":
            result["FX_POTENTIAL"] = fx_expected
        elif self.prediction_mode == "threshold":
            result["FX_POTENTIAL"] = fx_threshold
        elif self.prediction_mode == "soft_expected":
            result["FX_POTENTIAL"] = fx_soft_expected
        else:
            raise ValueError("prediction_mode має бути 'expected', 'threshold' або 'soft_expected'")

        return result
    







# =========================
# SAVE TWO-STAGE FX MODEL
# =========================

MODEL_PATH = r"C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Commissions_FX.pkl"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# окремі списки фіч для класифікації та регресії
CLF_FEATURES = list(X_train_clf.columns)   # тут FX_TYPE вже немає
REG_FEATURES = list(X_train_reg.columns)   # тут FX_TYPE є

# окремі категоріальні фічі
CLF_CAT_COLS = [
    c for c in CLF_FEATURES
    if str(X_train_clf[c].dtype) == "category"
]

REG_CAT_COLS = [
    c for c in REG_FEATURES
    if str(X_train_reg[c].dtype) == "category"
]

print("CLF features:", len(CLF_FEATURES))
print("REG features:", len(REG_FEATURES))

print("CLF categorical:", CLF_CAT_COLS)
print("REG categorical:", REG_CAT_COLS)

print("FX_TYPE in CLF:", "FX_TYPE" in CLF_FEATURES)
print("FX_TYPE in REG:", "FX_TYPE" in REG_FEATURES)


def collect_num_medians(df_train_source, features, cat_cols):
    num_cols = [c for c in features if c not in cat_cols]

    medians = {}

    for c in num_cols:
        if c in df_train_source.columns:
            medians[c] = pd.to_numeric(df_train_source[c], errors="coerce").median()
        else:
            medians[c] = 0

    return medians


def collect_category_values(X_train_source, cat_cols):
    category_values = {}

    for c in cat_cols:
        if c in X_train_source.columns:
            if str(X_train_source[c].dtype) == "category":
                cats = list(X_train_source[c].cat.categories.astype(str))
            else:
                cats = list(
                    X_train_source[c]
                    .astype("string")
                    .fillna("UNKNOWN")
                    .unique()
                )

            if "UNKNOWN" not in cats:
                cats.append("UNKNOWN")

            category_values[c] = cats

    return category_values


clf_num_medians = collect_num_medians(
    df_train_source=df_train,
    features=CLF_FEATURES,
    cat_cols=CLF_CAT_COLS
)

reg_num_medians = collect_num_medians(
    df_train_source=df_train_reg,
    features=REG_FEATURES,
    cat_cols=REG_CAT_COLS
)

clf_category_values = collect_category_values(
    X_train_source=X_train_clf,
    cat_cols=CLF_CAT_COLS
)

reg_category_values = collect_category_values(
    X_train_source=X_train_reg,
    cat_cols=REG_CAT_COLS
)


fx_model = TwoStageFXModel(
    clf_model=clf_binary,
    reg_model=reg,

    clf_features=CLF_FEATURES,
    reg_features=REG_FEATURES,

    clf_cat_cols=CLF_CAT_COLS,
    reg_cat_cols=REG_CAT_COLS,

    clf_num_medians=clf_num_medians,
    reg_num_medians=reg_num_medians,

    clf_category_values=clf_category_values,
    reg_category_values=reg_category_values,

    threshold=0.30,

    # рекомендую поки soft_expected, щоб класифікатор не прибивав великі суми
    prediction_mode="soft_expected"
)

joblib.dump(fx_model, MODEL_PATH)

print("Saved FX model to:")
print(MODEL_PATH)




























import joblib
import numpy as np
import pandas as pd

MODEL_PATH = r"C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Commissions_FX.pkl"

model = joblib.load(MODEL_PATH)


df_fx = model.predict_full(df)
df["FX_POTENTIAL"] = df_fx["FX_POTENTIAL"].values