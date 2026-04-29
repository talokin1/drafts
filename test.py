import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "expected_potential_curr_acc_model.pkl"


class ExpectedPotentialModel:
    def __init__(
        self,
        clf,
        reg,
        feature_cols,
        cat_cols,
        cat_values,
        bias_correction,
        zero_threshold,
        gamma,
        qm_blend,
        quantile_mapper,
        calibration_table,
        segment_calibration_table,
        global_calibration_factor,
        caps_by_segment,
        global_cap,
        active_threshold,
        segment_col,
        id_col,
        n_deciles
    ):
        self.clf = clf
        self.reg = reg
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.cat_values = cat_values

        self.bias_correction = bias_correction
        self.zero_threshold = zero_threshold
        self.gamma = gamma
        self.qm_blend = qm_blend
        self.quantile_mapper = quantile_mapper

        self.calibration_table = calibration_table
        self.segment_calibration_table = segment_calibration_table
        self.global_calibration_factor = global_calibration_factor

        self.caps_by_segment = caps_by_segment
        self.global_cap = global_cap

        self.active_threshold = active_threshold
        self.segment_col = segment_col
        self.id_col = id_col
        self.n_deciles = n_deciles

    def _prepare_X(self, df):
        X = df.copy()

        missing_cols = [c for c in self.feature_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for inference: {missing_cols}")

        X = X[self.feature_cols].copy()

        for c in self.cat_cols:
            if c in X.columns:
                X[c] = pd.Categorical(X[c], categories=self.cat_values[c])

        return X

    def _make_prediction_deciles(self, pred):
        pred = pd.Series(pred)

        try:
            return pd.qcut(
                pred,
                q=self.n_deciles,
                labels=False,
                duplicates="drop"
            )
        except ValueError:
            return pd.Series(np.zeros(len(pred), dtype=int), index=pred.index)

    def _soft_zero_multiplier(self, p, low, high):
        p = np.asarray(p)
        m = (p - low) / (high - low)
        return np.clip(m, 0, 1)

    def _apply_quantile_mapping(self, pred):
        pred = np.asarray(pred, dtype=float)
        out = pred.copy()

        mask = pred > self.active_threshold

        if mask.sum() > 0:
            mapped = np.expm1(
                self.quantile_mapper.predict(np.log1p(pred[mask]))
            )

            out[mask] = (
                (1 - self.qm_blend) * pred[mask]
                + self.qm_blend * mapped
            )

        return np.clip(out, 0, None)

    def _apply_calibration(self, X_part, pred_raw):
        tmp = pd.DataFrame(index=X_part.index)

        if self.segment_col in X_part.columns:
            tmp[self.segment_col] = X_part[self.segment_col].astype(str).values
        else:
            tmp[self.segment_col] = "ALL"

        tmp["pred_raw"] = pred_raw
        tmp["pred_decile"] = self._make_prediction_deciles(tmp["pred_raw"]).values

        tmp = tmp.merge(
            self.calibration_table[
                [self.segment_col, "pred_decile", "factor"]
            ],
            on=[self.segment_col, "pred_decile"],
            how="left"
        )

        tmp = tmp.merge(
            self.segment_calibration_table[
                [self.segment_col, "factor"]
            ].rename(columns={"factor": "segment_factor"}),
            on=self.segment_col,
            how="left"
        )

        tmp["factor"] = tmp["factor"].fillna(tmp["segment_factor"])
        tmp["factor"] = tmp["factor"].fillna(self.global_calibration_factor)
        tmp["factor"] = tmp["factor"].fillna(1.0)

        calibrated = pred_raw * tmp["factor"].values

        return calibrated, tmp["factor"].values

    def _apply_caps(self, X_part, pred):
        pred = np.asarray(pred, dtype=float)
        caps = np.full(len(pred), self.global_cap, dtype=float)

        if self.segment_col in X_part.columns:
            segments = X_part[self.segment_col].astype(str).values

            for i, seg in enumerate(segments):
                caps[i] = self.caps_by_segment.get(seg, self.global_cap)

        pred_capped = np.minimum(pred, caps)
        pred_capped = np.clip(pred_capped, 0, None)

        return pred_capped, caps

    def predict(self, df, return_details=True):
        X = self._prepare_X(df)

        p_active = self.clf.predict_proba(X)[:, 1]

        income_if_active_log = self.reg.predict(X)
        income_if_active = np.expm1(income_if_active_log)
        income_if_active = np.clip(income_if_active, 0, None)

        income_if_active = income_if_active * self.bias_correction

        expected_raw = (p_active ** self.gamma) * income_if_active

        soft_low = self.zero_threshold * 0.5
        soft_high = self.zero_threshold

        soft_mult = self._soft_zero_multiplier(
            p_active,
            soft_low,
            soft_high
        )

        expected_raw = expected_raw * soft_mult
        expected_raw = self._apply_quantile_mapping(expected_raw)

        expected_calibrated, calibration_factor = self._apply_calibration(
            X,
            expected_raw
        )

        final_pred, caps_used = self._apply_caps(
            X,
            expected_calibrated
        )

        if not return_details:
            return final_pred

        result = pd.DataFrame(index=df.index)

        if self.id_col in df.columns:
            result[self.id_col] = df[self.id_col].values

        if self.segment_col in df.columns:
            result[self.segment_col] = df[self.segment_col].astype(str).values

        result["P_ACTIVE"] = p_active
        result["IS_LIKELY_ACTIVE"] = (p_active >= self.zero_threshold).astype(int)
        result["Income_If_Active"] = income_if_active
        result["Expected_Raw"] = expected_raw
        result["Calibration_Factor"] = calibration_factor
        result["Expected_Calibrated"] = expected_calibrated
        result["Cap_Used"] = caps_used
        result["CURR_ACC_POTENTIAL"] = final_pred

        result["CURR_ACC_POTENTIAL"] = result["CURR_ACC_POTENTIAL"].round(2)
        result["Income_If_Active"] = result["Income_If_Active"].round(2)
        result["Expected_Raw"] = result["Expected_Raw"].round(2)
        result["Expected_Calibrated"] = result["Expected_Calibrated"].round(2)

        return result
    

expected_model = ExpectedPotentialModel(
    clf=clf,
    reg=reg,
    feature_cols=feature_cols,
    cat_cols=cat_cols,
    cat_values=cat_values,
    bias_correction=bias_correction,
    zero_threshold=ZERO_THRESHOLD,
    gamma=GAMMA,
    qm_blend=QM_BLEND,
    quantile_mapper=quantile_mapper,
    calibration_table=calibration_table,
    segment_calibration_table=segment_calibration_table,
    global_calibration_factor=global_calibration_factor,
    caps_by_segment=caps_by_segment,
    global_cap=global_cap,
    active_threshold=ACTIVE_THRESHOLD,
    segment_col=SEGMENT_COL,
    id_col=ID_COL,
    n_deciles=N_DECILES
)

joblib.dump(expected_model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")






import joblib

MODEL_PATH = "expected_potential_curr_acc_model.pkl"

model = joblib.load(MODEL_PATH)

# df_inference має бути вже препроцеснутий так само, як X для train
# але бажано з IDENTIFYCODE, якщо він потрібен для merge

predictions = model.predict(
    df_inference,
    return_details=True
)

predictions.head()

df_result = df_inference.copy()

predictions = model.predict(df_inference, return_details=True)

df_result["CURR_ACC_POTENTIAL"] = predictions["CURR_ACC_POTENTIAL"].values

df_result.head()