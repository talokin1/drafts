import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    r2_score
)


RANDOM_STATE = 42

TARGET_NAME = "CURR_ACC"             
ID_COL = "IDENTIFYCODE"
SEGMENT_COL = "FIRM_TYPE"

ACTIVE_THRESHOLD = 100

CLASSIFICATION_THRESHOLD = 0.5

N_DECILES = 10
MIN_GROUP_SIZE_FOR_CALIBRATION = 50
CALIBRATION_FACTOR_MIN = 0.25
CALIBRATION_FACTOR_MAX = 3.00

CAP_QUANTILE_BY_SEGMENT = 0.99

MODEL_PATH = "expected_potential_liabilities_model.pkl"

def prepare_categorical_train_valid(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    cat_cols = [
        c for c in X_train.columns
        if X_train[c].dtype.name in ("object", "category")
    ]

    cat_values = {}

    for c in cat_cols:
        train_categories = pd.Series(X_train[c].astype("object")).dropna().unique().tolist()
        cat_values[c] = train_categories

        X_train[c] = pd.Categorical(X_train[c], categories=train_categories)
        X_val[c] = pd.Categorical(X_val[c], categories=train_categories)

    return X_train, X_val, cat_cols, cat_values


def apply_categorical_inference(X_new, feature_cols, cat_cols, cat_values):
    X_new = X_new.copy()
    X_new = X_new[feature_cols].copy()

    for c in cat_cols:
        if c in X_new.columns:
            X_new[c] = pd.Categorical(X_new[c], categories=cat_values[c])

    return X_new


def make_prediction_deciles(pred, n_deciles=10):
    pred = pd.Series(pred)

    try:
        return pd.qcut(pred, q=n_deciles, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.zeros(len(pred), dtype=int), index=pred.index)


def safe_divide(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


df_model = X.copy()
y_clean = pd.Series(y, index=X.index).clip(lower=0)

# active target
y_active = (y_clean > ACTIVE_THRESHOLD).astype(int)

print("Target distribution:")
print(y_clean.describe())

print("\nActive rate:")
print(y_active.value_counts(normalize=True))
print(y_active.value_counts())

X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    df_model,
    y_clean,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_active
)

y_train_active = (y_train_raw > ACTIVE_THRESHOLD).astype(int)
y_val_active = (y_val_raw > ACTIVE_THRESHOLD).astype(int)

X_train, X_val, cat_cols, cat_values = prepare_categorical_train_valid(X_train, X_val)

feature_cols = X_train.columns.tolist()

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Categorical columns:", cat_cols)

def build_segment_weights(X_part, default_weight=1.0):
    weights = pd.Series(default_weight, index=X_part.index, dtype=float)

    if SEGMENT_COL in X_part.columns:
        weights.loc[X_part[SEGMENT_COL].astype(str).eq("MICRO")] = 1.0
        weights.loc[X_part[SEGMENT_COL].astype(str).eq("SMALL")] = 1.2
        weights.loc[X_part[SEGMENT_COL].astype(str).eq("MEDIUM")] = 1.5
        weights.loc[X_part[SEGMENT_COL].astype(str).eq("LARGE")] = 2.5

    return weights


clf_sample_weight = build_segment_weights(X_train)

# додатково можна підсилити активний клас, якщо дисбаланс сильний
pos_rate = y_train_active.mean()
neg_rate = 1 - pos_rate

if pos_rate > 0:
    class_weight_pos = neg_rate / pos_rate
else:
    class_weight_pos = 1.0

clf_sample_weight = clf_sample_weight * np.where(
    y_train_active == 1,
    class_weight_pos,
    1.0
)

print("Positive class weight multiplier:", class_weight_pos)




clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Training Stage 1: Activity Classifier...")

clf.fit(
    X_train,
    y_train_active,
    sample_weight=clf_sample_weight,
    eval_set=[(X_val, y_val_active)],
    eval_metric="auc",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

print("Training Stage 1: Activity Classifier...")

clf.fit(
    X_train,
    y_train_active,
    sample_weight=clf_sample_weight,
    eval_set=[(X_val, y_val_active)],
    eval_metric="auc",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)



mask_train_reg = (y_train_raw > ACTIVE_THRESHOLD).values
mask_val_reg = (y_val_raw > ACTIVE_THRESHOLD).values

X_train_reg = X_train.loc[mask_train_reg].copy()
X_val_reg = X_val.loc[mask_val_reg].copy()

y_train_reg_raw = y_train_raw.loc[mask_train_reg].copy()
y_val_reg_raw = y_val_raw.loc[mask_val_reg].copy()

y_train_reg_log = np.log1p(y_train_reg_raw)
y_val_reg_log = np.log1p(y_val_reg_raw)

print("Regression train shape:", X_train_reg.shape)
print("Regression val shape:", X_val_reg.shape)
print("Regression target describe:")
print(y_train_reg_raw.describe())


reg_sample_weight = build_segment_weights(X_train_reg)

# додатковий weight на великі income, але м'який
# щоб модель не ігнорувала хвіст, але й не розносила prediction
reg_sample_weight = reg_sample_weight * (1.0 + np.log1p(y_train_reg_raw) / np.log1p(y_train_reg_raw).max())


reg = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


print("Training Stage 2: Conditional Income Regressor...")

reg.fit(
    X_train_reg,
    y_train_reg_log,
    sample_weight=reg_sample_weight,
    eval_set=[(X_val_reg, y_val_reg_log)],
    eval_metric="mae",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

raw_expected = p_active * income_if_active
train_income_if_active_log = reg.predict(X_train)
val_income_if_active_log = reg.predict(X_val)

train_income_if_active = np.expm1(train_income_if_active_log)
val_income_if_active = np.expm1(val_income_if_active_log)

train_income_if_active = np.clip(train_income_if_active, 0, None)
val_income_if_active = np.clip(val_income_if_active, 0, None)

train_expected_raw = train_p_active * train_income_if_active
val_expected_raw = val_p_active * val_income_if_active

val_hard_gate_pred = np.where(
    val_p_active >= CLASSIFICATION_THRESHOLD,
    val_income_if_active,
    0
)




# CALIBRATION

def build_calibration_table(
    X_val,
    y_val_true,
    y_val_pred_raw,
    segment_col=SEGMENT_COL,
    n_deciles=10,
    min_group_size=50,
    factor_min=0.25,
    factor_max=3.00
):
    tmp = X_val[[segment_col]].copy() if segment_col in X_val.columns else pd.DataFrame(index=X_val.index)
    
    if segment_col not in tmp.columns:
        tmp[segment_col] = "ALL"

    tmp["true"] = y_val_true.values
    tmp["pred_raw"] = y_val_pred_raw
    tmp["pred_decile"] = make_prediction_deciles(tmp["pred_raw"], n_deciles=n_deciles).values

    # segment-level fallback
    segment_table = (
        tmp.groupby(segment_col)
        .agg(
            n=("true", "size"),
            true_sum=("true", "sum"),
            pred_sum=("pred_raw", "sum")
        )
        .reset_index()
    )

    segment_table["factor"] = segment_table.apply(
        lambda r: safe_divide(r["true_sum"], r["pred_sum"]),
        axis=1
    )

    global_factor = safe_divide(tmp["true"].sum(), tmp["pred_raw"].sum())

    if pd.isna(global_factor):
        global_factor = 1.0

    segment_table["factor"] = segment_table["factor"].fillna(global_factor)
    segment_table["factor"] = segment_table["factor"].clip(factor_min, factor_max)

    # segment × decile table
    group_table = (
        tmp.groupby([segment_col, "pred_decile"])
        .agg(
            n=("true", "size"),
            true_sum=("true", "sum"),
            pred_sum=("pred_raw", "sum"),
            true_mean=("true", "mean"),
            pred_mean=("pred_raw", "mean"),
            true_median=("true", "median"),
            pred_median=("pred_raw", "median")
        )
        .reset_index()
    )

    group_table["factor"] = group_table.apply(
        lambda r: safe_divide(r["true_sum"], r["pred_sum"]),
        axis=1
    )

    group_table = group_table.merge(
        segment_table[[segment_col, "factor"]].rename(columns={"factor": "segment_factor"}),
        on=segment_col,
        how="left"
    )

    group_table["factor"] = np.where(
        group_table["n"] >= min_group_size,
        group_table["factor"],
        group_table["segment_factor"]
    )

    group_table["factor"] = group_table["factor"].fillna(global_factor)
    group_table["factor"] = group_table["factor"].clip(factor_min, factor_max)

    return group_table, segment_table, global_factor



calibration_table, segment_calibration_table, global_calibration_factor = build_calibration_table(
    X_val=X_val,
    y_val_true=y_val_raw,
    y_val_pred_raw=val_expected_raw,
    segment_col=SEGMENT_COL,
    n_deciles=N_DECILES,
    min_group_size=MIN_GROUP_SIZE_FOR_CALIBRATION,
    factor_min=CALIBRATION_FACTOR_MIN,
    factor_max=CALIBRATION_FACTOR_MAX
)

print("Global calibration factor:", global_calibration_factor)
display(calibration_table.head(20))
display(segment_calibration_table)


def apply_calibration(
    X_part,
    pred_raw,
    calibration_table,
    segment_calibration_table,
    global_factor,
    segment_col=SEGMENT_COL,
    n_deciles=10
):
    tmp = pd.DataFrame(index=X_part.index)
    
    if segment_col in X_part.columns:
        tmp[segment_col] = X_part[segment_col].astype(str).values
    else:
        tmp[segment_col] = "ALL"

    tmp["pred_raw"] = pred_raw
    tmp["pred_decile"] = make_prediction_deciles(tmp["pred_raw"], n_deciles=n_deciles).values

    tmp = tmp.merge(
        calibration_table[[segment_col, "pred_decile", "factor"]],
        on=[segment_col, "pred_decile"],
        how="left"
    )

    tmp = tmp.merge(
        segment_calibration_table[[segment_col, "factor"]].rename(columns={"factor": "segment_factor"}),
        on=segment_col,
        how="left"
    )

    tmp["factor"] = tmp["factor"].fillna(tmp["segment_factor"])
    tmp["factor"] = tmp["factor"].fillna(global_factor)
    tmp["factor"] = tmp["factor"].fillna(1.0)

    calibrated = pred_raw * tmp["factor"].values

    return calibrated, tmp["factor"].values

train_expected_calibrated, train_calibration_factor = apply_calibration(
    X_train,
    train_expected_raw,
    calibration_table,
    segment_calibration_table,
    global_calibration_factor,
    segment_col=SEGMENT_COL,
    n_deciles=N_DECILES
)

val_expected_calibrated, val_calibration_factor = apply_calibration(
    X_val,
    val_expected_raw,
    calibration_table,
    segment_calibration_table,
    global_calibration_factor,
    segment_col=SEGMENT_COL,
    n_deciles=N_DECILES
)



def build_caps_by_segment(
    X_train,
    y_train,
    segment_col=SEGMENT_COL,
    active_threshold=ACTIVE_THRESHOLD,
    cap_quantile=0.99
):
    tmp = X_train[[segment_col]].copy() if segment_col in X_train.columns else pd.DataFrame(index=X_train.index)

    if segment_col not in tmp.columns:
        tmp[segment_col] = "ALL"

    tmp["target"] = y_train.values

    tmp_active = tmp[tmp["target"] > active_threshold].copy()

    caps = (
        tmp_active.groupby(segment_col)["target"]
        .quantile(cap_quantile)
        .to_dict()
    )

    global_cap = tmp_active["target"].quantile(cap_quantile)

    if pd.isna(global_cap):
        global_cap = y_train.quantile(cap_quantile)

    return caps, global_cap

caps_by_segment, global_cap = build_caps_by_segment(
    X_train=X_train,
    y_train=y_train_raw,
    segment_col=SEGMENT_COL,
    active_threshold=ACTIVE_THRESHOLD,
    cap_quantile=CAP_QUANTILE_BY_SEGMENT
)

print("Global cap:", global_cap)
print("Caps by segment:")
print(caps_by_segment)


def apply_caps(
    X_part,
    pred,
    caps_by_segment,
    global_cap,
    segment_col=SEGMENT_COL
):
    pred = np.array(pred, dtype=float)
    caps = np.full(len(pred), global_cap, dtype=float)

    if segment_col in X_part.columns:
        segments = X_part[segment_col].astype(str).values

        for i, seg in enumerate(segments):
            caps[i] = caps_by_segment.get(seg, global_cap)

    pred_capped = np.minimum(pred, caps)
    pred_capped = np.clip(pred_capped, 0, None)

    return pred_capped, caps


train_final_pred, train_caps_used = apply_caps(
    X_train,
    train_expected_calibrated,
    caps_by_segment,
    global_cap,
    segment_col=SEGMENT_COL
)

val_final_pred, val_caps_used = apply_caps(
    X_val,
    val_expected_calibrated,
    caps_by_segment,
    global_cap,
    segment_col=SEGMENT_COL
)




def regression_report(y_true, y_pred, title):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.clip(y_pred, 0, None))

    eps = 1e-9

    report = {
        "title": title,
        "n": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAE_log": mean_absolute_error(y_true_log, y_pred_log),
        "MedAE_log": median_absolute_error(y_true_log, y_pred_log),
        "R2_log": r2_score(y_true_log, y_pred_log),
        "sum_true": y_true.sum(),
        "sum_pred": y_pred.sum(),
        "sum_ratio": y_pred.sum() / max(y_true.sum(), eps),
        "mean_true": y_true.mean(),
        "mean_pred": y_pred.mean(),
        "bias_mean": np.mean(y_pred - y_true),
        "bias_sum": y_pred.sum() - y_true.sum(),
        "overprediction_rate": np.mean(y_pred > y_true),
        "underprediction_rate": np.mean(y_pred < y_true),
    }

    print("=" * 80)
    print(title)
    print("-" * 80)
    for k, v in report.items():
        if k != "title":
            print(f"{k}: {v}")
    print("=" * 80)

    return report

train_report_raw = regression_report(
    y_train_raw,
    train_expected_raw,
    "[Train] Raw Expected = P_ACTIVE * INCOME_IF_ACTIVE"
)

val_report_raw = regression_report(
    y_val_raw,
    val_expected_raw,
    "[Validation] Raw Expected = P_ACTIVE * INCOME_IF_ACTIVE"
)

val_report_hard = regression_report(
    y_val_raw,
    val_hard_gate_pred,
    "[Validation] Old-style Hard Gate Prediction"
)

train_report_final = regression_report(
    y_train_raw,
    train_final_pred,
    "[Train] Final Expected Potential after Calibration + Caps"
)

val_report_final = regression_report(
    y_val_raw,
    val_final_pred,
    "[Validation] Final Expected Potential after Calibration + Caps"
)


validation_results = pd.DataFrame({
    ID_COL: X_val.index,
    "True_Value": y_val_raw.values,
    "P_ACTIVE": val_p_active,
    "IS_LIKELY_ACTIVE": (val_p_active >= CLASSIFICATION_THRESHOLD).astype(int),
    "Income_If_Active": val_income_if_active,
    "Expected_Raw": val_expected_raw,
    "Calibration_Factor": val_calibration_factor,
    "Expected_Calibrated": val_expected_calibrated,
    "Cap_Used": val_caps_used,
    "Predicted": val_final_pred
})

if SEGMENT_COL in X_val.columns:
    validation_results[SEGMENT_COL] = X_val[SEGMENT_COL].astype(str).values

validation_results["Error"] = validation_results["Predicted"] - validation_results["True_Value"]
validation_results["Abs_Error"] = validation_results["Error"].abs()
validation_results["Ratio"] = validation_results["Predicted"] / validation_results["True_Value"].replace(0, np.nan)

validation_results["Predicted"] = validation_results["Predicted"].round(2)
validation_results["Expected_Raw"] = validation_results["Expected_Raw"].round(2)
validation_results["Expected_Calibrated"] = validation_results["Expected_Calibrated"].round(2)
validation_results["Income_If_Active"] = validation_results["Income_If_Active"].round(2)

validation_results.head(20)


validation_results = pd.DataFrame({
    ID_COL: X_val.index,
    "True_Value": y_val_raw.values,
    "P_ACTIVE": val_p_active,
    "IS_LIKELY_ACTIVE": (val_p_active >= CLASSIFICATION_THRESHOLD).astype(int),
    "Income_If_Active": val_income_if_active,
    "Expected_Raw": val_expected_raw,
    "Calibration_Factor": val_calibration_factor,
    "Expected_Calibrated": val_expected_calibrated,
    "Cap_Used": val_caps_used,
    "Predicted": val_final_pred
})

if SEGMENT_COL in X_val.columns:
    validation_results[SEGMENT_COL] = X_val[SEGMENT_COL].astype(str).values

validation_results["Error"] = validation_results["Predicted"] - validation_results["True_Value"]
validation_results["Abs_Error"] = validation_results["Error"].abs()
validation_results["Ratio"] = validation_results["Predicted"] / validation_results["True_Value"].replace(0, np.nan)

validation_results["Predicted"] = validation_results["Predicted"].round(2)
validation_results["Expected_Raw"] = validation_results["Expected_Raw"].round(2)
validation_results["Expected_Calibrated"] = validation_results["Expected_Calibrated"].round(2)
validation_results["Income_If_Active"] = validation_results["Income_If_Active"].round(2)

validation_results.head(20)

if SEGMENT_COL in validation_results.columns:
    segment_report = group_diagnostics(validation_results, [SEGMENT_COL])
    display(segment_report)


validation_results["pred_decile"] = pd.qcut(
    validation_results["Predicted"],
    q=10,
    labels=False,
    duplicates="drop"
)

decile_report = group_diagnostics(validation_results, ["pred_decile"])
display(decile_report)


if SEGMENT_COL in validation_results.columns:
    segment_decile_report = group_diagnostics(validation_results, [SEGMENT_COL, "pred_decile"])
    display(segment_decile_report)


validation_results.sort_values("Error", ascending=False).head(30)
validation_results.sort_values("Error", ascending=True).head(30)


validation_results[
    (validation_results["True_Value"] > ACTIVE_THRESHOLD) &
    (validation_results["Predicted"] <= ACTIVE_THRESHOLD)
].sort_values("True_Value", ascending=False).head(30)


validation_results[
    (validation_results["True_Value"] > ACTIVE_THRESHOLD) &
    (validation_results["Predicted"] <= ACTIVE_THRESHOLD)
].sort_values("True_Value", ascending=False).head(30)

plt.figure(figsize=(10, 5))

sns.kdeplot(np.log1p(validation_results["True_Value"]), label="True", fill=True, alpha=0.2)
sns.kdeplot(np.log1p(validation_results["Predicted"]), label="Predicted", linestyle="--")

plt.xlabel("log1p income")
plt.title("Distribution Match: True vs Final Prediction")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))

sns.kdeplot(np.log1p(validation_results["True_Value"]), label="True", fill=True, alpha=0.2)
sns.kdeplot(np.log1p(validation_results["Predicted"]), label="Predicted", linestyle="--")

plt.xlabel("log1p income")
plt.title("Distribution Match: True vs Final Prediction")
plt.legend()
plt.show()



