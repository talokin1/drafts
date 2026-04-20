
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_feature_imp(model, feature_names):
    return pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': feature_names
    }).sort_values(by="Value", ascending=False)

clf_imp = get_feature_imp(clf, X_train.columns)
reg_imp = get_feature_imp(reg, X_train.columns)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.barplot(x="Value", y="Feature", data=clf_imp.head(30), ax=axes[0], palette="viridis")
axes[0].set_title('Top 30 Features: Classifier (Probability of Active)')

sns.barplot(x="Value", y="Feature", data=reg_imp.head(30), ax=axes[1], palette="magma")
axes[1].set_title('Top 30 Features: Regressor (Log Amount)')

plt.tight_layout()
plt.show()

zero_clf = clf_imp[clf_imp['Value'] == 0]['Feature'].tolist()
zero_reg = reg_imp[reg_imp['Value'] == 0]['Feature'].tolist()
print(f"Useless features for Classifier: {len(zero_clf)}")
print(f"Useless features for Regressor : {len(zero_reg)}")




validation_results = pd.DataFrame({
    'IDENTIFYCODE': X_val.index,
    'True_Value': y_val_raw.values,
    'Predicted': y_pred_final
})

validation_results['Predicted'] = validation_results['Predicted'].round(2)
display(validation_results.head(10))



import joblib
import numpy as np

class TwoStageLiabilitiesModel:
    def __init__(self, classifier, regressor, cat_cols, feature_cols, threshold):
        self.classifier = classifier
        self.regressor = regressor
        self.cat_cols = cat_cols
        self.feature_cols = feature_cols
        self.threshold = threshold 
        
    def predict(self, X):
        X_pred = X.copy()
        
        X_pred = X_pred[self.feature_cols]
        
        for c in self.cat_cols:
            X_pred[c] = X_pred[c].astype("category")
            
        class_preds = self.classifier.predict(X_pred)
        
        reg_preds_log = self.regressor.predict(X_pred)
        
        reg_preds = np.expm1(reg_preds_log)
        
        final_predictions = np.where(class_preds == 1, reg_preds, 0)
        
        return final_predictions

two_stage_model = TwoStageLiabilitiesModel(
    classifier=clf, 
    regressor=reg, 
    cat_cols=cat_cols, 
    feature_cols=X_train.columns.to_list(),
    threshold=THRESHOLD # 400 у твоєму поточному вдалому експерименті
)

save_path = r"C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Liabilities_TwoStage.pkl"
joblib.dump(two_stage_model, save_path)
print(f"Модель успішно збережена за шляхом:\n{save_path}")

# loaded_model = joblib.load(save_path)
# test_predictions = loaded_model.predict(new_data_dataframe)