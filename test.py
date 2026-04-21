import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix

def find_optimal_threshold(y_true, y_prob, target_recall=0.95, beta=2.0, plot=True):
    """
    Автоматично підбирає оптимальний поріг класифікації для максимізації Recall.
    
    Параметри:
    - y_true: реальні мітки класів (0 або 1)
    - y_prob: ймовірності класу 1, передбачені моделлю
    - target_recall: цільове значення Recall (для Стратегії 2)
    - beta: вага для F-beta score (beta=2 означає, що Recall вдвічі важливіший)
    """
    # Обчислюємо значення для кривої
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Відкидаємо останній елемент, оскільки thresholds має на 1 елемент менше
    p = precision[:-1]
    r = recall[:-1]
    
    # СТРАТЕГІЯ 1: Максимізація F-beta score
    # Додаємо 1e-10, щоб уникнути ділення на нуль
    f_beta_scores = ((1 + beta**2) * (p * r)) / ((beta**2 * p) + r + 1e-10)
    best_fbeta_idx = np.argmax(f_beta_scores)
    thresh_fbeta = thresholds[best_fbeta_idx]
    
    # СТРАТЕГІЯ 2: Максимальний поріг при заданому мінімальному Recall
    # Шукаємо найвищий можливий поріг, який все ще забезпечує Recall >= target_recall
    valid_indices = np.where(r >= target_recall)[0]
    if len(valid_indices) > 0:
        best_recall_idx = valid_indices[-1] # Беремо останній (найвищий) поріг з підходящих
        thresh_recall = thresholds[best_recall_idx]
    else:
        thresh_recall = thresholds[0] # Якщо неможливо досягти, беремо найменший поріг
        
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, p, label='Precision', linestyle='--')
        plt.plot(thresholds, r, label='Recall', linestyle='-')
        plt.plot(thresholds, f_beta_scores, label=f'F{beta} Score', color='green', linewidth=2)
        
        plt.axvline(thresh_fbeta, color='green', linestyle=':', 
                    label=f'Opt F{beta} Thresh: {thresh_fbeta:.3f}')
        plt.axvline(thresh_recall, color='red', linestyle=':', 
                    label=f'Min {target_recall*100}% Recall Thresh: {thresh_recall:.3f}')
        
        plt.title('Precision, Recall & F-beta by Threshold')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    return {
        'threshold_fbeta': thresh_fbeta,
        'threshold_target_recall': thresh_recall
    }

# ==========================================
# ЯК ІНТЕГРУВАТИ ЦЕ В ТВІЙ КОД (після clf.fit)
# ==========================================

print("Оцінка ймовірностей на валідації...")
probs_val = clf.predict_proba(X_val)[:, 1]

print("Підбір оптимального порогу...")
# Знаходимо пороги (графік виведеться автоматично)
thresholds_dict = find_optimal_threshold(y_val_clf, probs_val, target_recall=0.98, beta=2.0)

# Вибираємо стратегію. Для оцінки потенціалу раджу жорстку фіксацію Recall (напр. 98%)
optimal_threshold = thresholds_dict['threshold_target_recall']
print(f"\nЗастосований поріг: {optimal_threshold:.4f}")

# Робимо нові класифікаційні передбачення
val_class_preds_optimized = (probs_val >= optimal_threshold).astype(int)

# Перевіряємо матрицю помилок до і після
print("\nМатриця помилок (Поріг 0.5):")
print(confusion_matrix(y_val_clf, (probs_val >= 0.5).astype(int)))

print(f"\nМатриця помилок (Оптимізований поріг {optimal_threshold:.4f}):")
print(confusion_matrix(y_val_clf, val_class_preds_optimized))

# Далі використовуй val_class_preds_optimized для об'єднання з регресією (y_pred_final)