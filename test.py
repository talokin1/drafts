import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report

cat_cols = [c for c in X_train_clf.columns if X_train_clf[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train_clf[c] = X_train_clf[c].astype("category")
    X_val_clf[c] = X_val_clf[c].astype("category")

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf.fit(
    X_train_clf,
    y_train_clf,
    eval_set=[(X_val_clf, y_val_clf)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=True)]
)

# оцінка
y_pred_proba = clf.predict_proba(X_val_clf)[:, 1]
print("ROC-AUC:", roc_auc_score(y_val_clf, y_pred_proba))
print(classification_report(y_val_clf, (y_pred_proba > 0.5).astype(int)))



cat_cols = [c for c in X_train_reg.columns if X_train_reg[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train_reg[c] = X_train_reg[c].astype("category")
    X_val_reg[c] = X_val_reg[c].astype("category")

reg = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

reg.fit(
    X_train_reg,
    y_train_reg,
    eval_set=[(X_val_reg, y_val_reg)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(200, verbose=True)]
)

y_val_proba = clf.predict_proba(X_val_clf)[:, 1]
y_val_pred = (y_val_proba > 0.5).astype(int)

clf_report = pd.DataFrame({
    "IDENTIFYCODE": df_val.index,
    "y_true": y_val_clf,
    "y_pred": y_val_pred,
    "proba": y_val_proba
})

print("ROC-AUC:", roc_auc_score(y_val_clf, y_val_proba))
print(classification_report(y_val_clf, y_val_pred))





y_pred_val = reg.predict(X_val_reg)

reg_report = pd.DataFrame({
    "IDENTIFYCODE": df_val_reg.index,
    "True": np.expm1(y_val_reg),
    "Pred": np.expm1(y_pred_val)
})


# 1. класифікація на всьому val
val_proba = clf.predict_proba(X_val_clf)[:, 1]
val_is_profitable = val_proba > 0.5

# 2. регресія тільки там, де треба
val_preds = np.zeros(len(X_val_clf))

mask = val_is_profitable
if mask.sum() > 0:
    val_preds[mask] = reg.predict(X_val_clf[mask])

# 3. повертаємо з логів якщо треба
val_preds_final = np.expm1(val_preds)

# 4. істинні значення
y_val_true = df_val[TARGET_NAME].values



validation_results = pd.DataFrame({
    "IDENTIFYCODE": df_val.index,
    "True": y_val_true,
    "Predicted": val_preds_final,
    "Proba": val_proba,
    "Is_Profitable_Pred": val_is_profitable.astype(int),
    "Is_Profitable_True": (y_val_true > 0).astype(int)
})








class TwoStageFXModel:
    def __init__(self, clf_model, reg_model, cat_cols, features_cols, threshold=0.5):
        self.clf_model = clf_model
        self.reg_model = reg_model
        self.cat_cols = cat_cols
        self.features_cols = features_cols
        self.threshold = threshold

    def _prepare_X(self, X):
        X = X.copy()
        X = X[self.features_cols]

        for c in self.cat_cols:
            X[c] = X[c].astype("category")

        return X

    def predict(self, X):
        X_prep = self._prepare_X(X)

        # 1. класифікація
        proba = self.clf_model.predict_proba(X_prep)[:, 1]
        is_profitable = proba > self.threshold

        # 2. регресія тільки для прибуткових
        preds = np.zeros(len(X_prep))

        if is_profitable.sum() > 0:
            preds[is_profitable] = self.reg_model.predict(X_prep[is_profitable])

        return preds, proba
    
import joblib

model = TwoStageFXModel(
    clf_model=clf,
    reg_model=reg,
    cat_cols=cat_cols,
    features_cols=final_features,
    threshold=0.5
)

joblib.dump(model, "fx_two_stage_model.pkl")



model = joblib.load("fx_two_stage_model.pkl")

df_full_proc, _ = build_features(df_full, fitted_params)

preds, proba = model.predict(df_full_proc)

df_result = pd.DataFrame({
    "IDENTIFYCODE": df_full.index,
    "FX_POTENTIAL": preds,
    "FX_PROBA": proba
})