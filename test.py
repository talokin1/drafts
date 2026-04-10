y_train_clf = (df_train[TARGET_NAME] > 0).astype(int)
y_val_clf   = (df_val[TARGET_NAME] > 0).astype(int)

X_train_clf = df_train_proc[final_features]
X_val_clf   = df_val_proc[final_features]



from sklearn.model_selection import train_test_split

X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)


cat_cols = [c for c in X_train_clf.columns if X_train_clf[c].dtype.name in ("object", "category")]

for c in cat_cols:
    X_train_clf[c] = X_train_clf[c].astype("category")
    X_val_clf[c] = X_val_clf[c].astype("category")

    X_train_reg[c] = X_train_reg[c].astype("category")
    X_val_reg[c] = X_val_reg[c].astype("category")



import lightgbm as lgb

clf_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)

clf_model.fit(
    X_train_clf, y_train_clf,
    eval_set=[(X_val_clf, y_val_clf)],
    eval_metric='auc',
    verbose=100
)



reg_model = lgb.LGBMRegressor(
    objective='huber',   # як у тебе
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)

reg_model.fit(
    X_train_reg, y_train_reg,
    eval_set=[(X_val_reg, y_val_reg)],
    eval_metric='mae',
    verbose=100
)



class TwoStageModel:
    def __init__(self, clf_model, reg_model, cat_cols, features, threshold=0.5):
        self.clf_model = clf_model
        self.reg_model = reg_model
        self.cat_cols = cat_cols
        self.features = features
        self.threshold = threshold

    def _prepare_X(self, X):
        X = X[self.features].copy()
        for c in self.cat_cols:
            X[c] = X[c].astype("category")
        return X

    def predict(self, X):
        X = self._prepare_X(X)

        # 1. Класифікація
        proba = self.clf_model.predict_proba(X)[:, 1]
        is_positive = proba > self.threshold

        # 2. Регресія
        preds = np.zeros(len(X))
        preds[is_positive] = self.reg_model.predict(X[is_positive])

        return preds, proba
    

model = TwoStageModel(
    clf_model=clf_model,
    reg_model=reg_model,
    cat_cols=cat_cols,
    features=final_features,
    threshold=0.3   # можеш тюнити
)

preds, proba = model.predict(df_full)

df_full["POTENTIAL"] = preds
df_full["PROBA"] = proba

import joblib

joblib.dump(model, "two_stage_model.pkl")