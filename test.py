import os
import joblib
import numpy as np
import pandas as pd


class HardFXTwoStageModel:
    """
    HARD FX two-stage model:

    1) Classifier predicts FX active / inactive.
    2) Regressor predicts FX amount conditional on activity.
    3) Final prediction:
        if FX_ACTIVE_PRED == 0 -> FX_FINAL_PRED = 0
        if FX_ACTIVE_PRED == 1 -> FX_FINAL_PRED = FX_COND_PRED
    """

    def __init__(
        self,
        clf_binary,
        calibrator,
        reg,
        clf_features,
        reg_features,
        clf_reference_df,
        reg_reference_df,
        classification_threshold,
        fx_upper_cap,
        active_threshold_target=100,
        target_name="FX",
        prepare_func=None
    ):
        self.clf_binary = clf_binary
        self.calibrator = calibrator
        self.reg = reg

        self.clf_features = clf_features
        self.reg_features = reg_features

        self.clf_reference_df = clf_reference_df.copy()
        self.reg_reference_df = reg_reference_df.copy()

        self.classification_threshold = classification_threshold
        self.fx_upper_cap = fx_upper_cap
        self.active_threshold_target = active_threshold_target
        self.target_name = target_name

        # prepare_train_val_X краще передавати явно або мати в notebook/global scope
        self.prepare_func = prepare_func

    def _get_prepare_func(self):
        if self.prepare_func is not None:
            return self.prepare_func

        if "prepare_train_val_X" in globals():
            return globals()["prepare_train_val_X"]

        raise ValueError(
            "prepare_train_val_X is not available. "
            "Define prepare_train_val_X before calling predict(), "
            "or pass prepare_func=prepare_train_val_X when creating the model."
        )

    def _check_features(self, df):
        missing_clf = [c for c in self.clf_features if c not in df.columns]
        missing_reg = [c for c in self.reg_features if c not in df.columns]

        if len(missing_clf) > 0:
            raise ValueError(f"Missing classification features: {missing_clf}")

        if len(missing_reg) > 0:
            raise ValueError(f"Missing regression features: {missing_reg}")

    def predict(self, df, return_full=True, id_cols=None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Inference dataframe.

        return_full : bool
            If True, returns id columns + all model outputs.
            If False, returns only FX_FINAL_PRED as numpy array.

        id_cols : list or None
            Columns to include in output. If None, the method tries common ID columns.

        Returns
        -------
        pd.DataFrame or np.ndarray
        """

        prepare_func = self._get_prepare_func()

        df_inf = df.copy()
        self._check_features(df_inf)

        # ====================================================
        # 1. Classification
        # ====================================================

        _, X_inf_clf, _ = prepare_func(
            df_train=self.clf_reference_df,
            df_val=df_inf,
            features=self.clf_features
        )

        p_fx_raw = self.clf_binary.predict_proba(X_inf_clf)[:, 1]
        p_fx = self.calibrator.predict(p_fx_raw)

        fx_active_pred = (
            p_fx >= self.classification_threshold
        ).astype(int)

        # ====================================================
        # 2. Conditional regression
        # ====================================================

        _, X_inf_reg, _ = prepare_func(
            df_train=self.reg_reference_df,
            df_val=df_inf,
            features=self.reg_features
        )

        pred_log_cond = self.reg.predict(X_inf_reg)
        pred_log_cond = np.clip(pred_log_cond, 0, None)

        fx_cond_pred = np.expm1(pred_log_cond)
        fx_cond_pred = np.clip(fx_cond_pred, 0, self.fx_upper_cap)

        # ====================================================
        # 3. HARD final logic
        # ====================================================

        fx_final_pred = np.where(
            fx_active_pred == 1,
            fx_cond_pred,
            0.0
        )

        fx_expected_soft = p_fx * fx_cond_pred

        if not return_full:
            return fx_final_pred

        # ====================================================
        # 4. Result dataframe
        # ====================================================

        if id_cols is None:
            id_cols = [
                c for c in [
                    "CONTRAGENTID",
                    "IDENTIFYCODE",
                    "FIRM_NAME",
                    "OKPO",
                    "CLIENT_ID"
                ]
                if c in df_inf.columns
            ]
        else:
            id_cols = [c for c in id_cols if c in df_inf.columns]

        result = pd.DataFrame(index=df_inf.index)

        if len(id_cols) > 0:
            result = pd.concat(
                [
                    df_inf[id_cols].copy(),
                    result
                ],
                axis=1
            )

        result["P_FX_RAW"] = p_fx_raw
        result["P_FX"] = p_fx
        result["FX_ACTIVE_PRED"] = fx_active_pred
        result["FX_COND_PRED"] = fx_cond_pred
        result["FX_EXPECTED_SOFT"] = fx_expected_soft
        result["FX_FINAL_PRED"] = fx_final_pred

        if self.target_name in df_inf.columns:
            result["FX_TRUE"] = pd.to_numeric(
                df_inf[self.target_name],
                errors="coerce"
            ).fillna(0)

            result["FX_ACTIVE_TRUE"] = (
                result["FX_TRUE"] > self.active_threshold_target
            ).astype(int)

        return result

    def predict_amount(self, df):
        """
        Returns only final HARD FX prediction.
        """
        return self.predict(df, return_full=False)

    def save(self, path):
        """
        Saves model to pkl.
        """
        model_to_save = {
            "clf_binary": self.clf_binary,
            "calibrator": self.calibrator,
            "reg": self.reg,

            "clf_features": self.clf_features,
            "reg_features": self.reg_features,

            "clf_reference_df": self.clf_reference_df,
            "reg_reference_df": self.reg_reference_df,

            "classification_threshold": self.classification_threshold,
            "fx_upper_cap": self.fx_upper_cap,
            "active_threshold_target": self.active_threshold_target,
            "target_name": self.target_name
        }

        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        joblib.dump(model_to_save, path)

        print("Model saved to:", path)

    @classmethod
    def load(cls, path, prepare_func=None):
        """
        Loads model from pkl.
        """
        artifact = joblib.load(path)

        model = cls(
            clf_binary=artifact["clf_binary"],
            calibrator=artifact["calibrator"],
            reg=artifact["reg"],

            clf_features=artifact["clf_features"],
            reg_features=artifact["reg_features"],

            clf_reference_df=artifact["clf_reference_df"],
            reg_reference_df=artifact["reg_reference_df"],

            classification_threshold=artifact["classification_threshold"],
            fx_upper_cap=artifact["fx_upper_cap"],
            active_threshold_target=artifact["active_threshold_target"],
            target_name=artifact["target_name"],

            prepare_func=prepare_func
        )

        print("Model loaded from:", path)

        return model
    


fx_model = HardFXTwoStageModel(
    clf_binary=clf_binary,
    calibrator=calibrator,
    reg=reg,

    clf_features=clf_features,
    reg_features=reg_features,

    clf_reference_df=df_clf_fit[clf_features].copy(),
    reg_reference_df=df_train_reg[reg_features].copy(),

    classification_threshold=CLASSIFICATION_THRESHOLD,
    fx_upper_cap=FX_UPPER_CAP,
    active_threshold_target=ACTIVE_THRESHOLD_TARGET,
    target_name=TARGET_NAME,

    prepare_func=prepare_train_val_X
)


# ============================================================
# CREATE HARD FX MODEL OBJECT
# ============================================================

fx_model = HardFXTwoStageModel(
    clf_binary=clf_binary,
    calibrator=calibrator,
    reg=reg,

    clf_features=clf_features,
    reg_features=reg_features,

    clf_reference_df=df_clf_fit[clf_features].copy(),
    reg_reference_df=df_train_reg[reg_features].copy(),

    classification_threshold=CLASSIFICATION_THRESHOLD,
    fx_upper_cap=FX_UPPER_CAP,
    active_threshold_target=ACTIVE_THRESHOLD_TARGET,
    target_name=TARGET_NAME,

    prepare_func=prepare_train_val_X
)


validation_results = fx_model.predict(df_val)

validation_results.head()

from sklearn.metrics import mean_absolute_error, median_absolute_error

y_true = validation_results["FX_TRUE"].values
y_pred = validation_results["FX_FINAL_PRED"].values

true_sum = y_true.sum()
pred_sum = y_pred.sum()

print("=" * 70)
print("HARD FX MODEL VALIDATION")
print("=" * 70)
print("Rows:", len(validation_results))
print("True active clients:", int((y_true > ACTIVE_THRESHOLD_TARGET).sum()))
print("Pred active clients:", int((y_pred > 0).sum()))
print("-" * 70)
print("MAE:", round(mean_absolute_error(y_true, y_pred), 2))
print("MedAE:", round(median_absolute_error(y_true, y_pred), 2))
print("-" * 70)
print("True sum:", round(true_sum, 2))
print("Pred sum:", round(pred_sum, 2))
print("Pred / true:", round(pred_sum / max(true_sum, 1), 4))

print("\nFX_FINAL_PRED quantiles:")
print(validation_results["FX_FINAL_PRED"].quantile([0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]))

print("\nP_FX quantiles:")
print(validation_results["P_FX"].quantile([0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]))





MODEL_PATH = "fx_hard_two_stage_model/fx_hard_two_stage_model.pkl"

fx_model.save(MODEL_PATH)




































MODEL_PATH = "fx_hard_two_stage_model/fx_hard_two_stage_model.pkl"

fx_model_loaded = HardFXTwoStageModel.load(
    MODEL_PATH,
    prepare_func=prepare_train_val_X
)
df_inference["FX_POTENTIAL"] = fx_model_loaded.predict_amount(df_inference)