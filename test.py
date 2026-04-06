y_pred_clf = clf.predict(X_val_clf)
y_pred_proba_clf = clf.predict_proba(X_val_clf)[:, 1]

print("\n" + "="*40)
print("МЕТРИКИ КЛАСИФІКАТОРА")
print("="*40)

# ROC-AUC показує загальну здатність моделі ранжувати класи (особливо важливо при дисбалансі)
roc_auc = roc_auc_score(y_val_clf, y_pred_proba_clf)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")

print("Confusion Matrix (Матриця помилок):")
# Формат: [[True Negative, False Positive],
#          [False Negative, True Positive]]
print(confusion_matrix(y_val_clf, y_pred_clf))
print("\nClassification Report:")
print(classification_report(y_val_clf, y_pred_clf))
