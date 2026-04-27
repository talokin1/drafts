from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Рахуємо значення P, R та пороги
precisions, recalls, thresholds = precision_recall_curve(y_val_clf, val_class_proba)

# Масив thresholds на 1 елемент коротший за precisions/recalls, 
# тому відкидаємо останній елемент для побудови графіків
precisions = precisions[:-1]
recalls = recalls[:-1]

# 2. Рахуємо F1-score для кожного порогу (щоб знайти математичний баланс)
# Додаємо eps, щоб уникнути ділення на нуль
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_idx]

# 3. Будуємо графік
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Лінія Precision
plt.plot(thresholds, precisions, label='Precision (Точність)', color='blue', linewidth=2)
# Лінія Recall
plt.plot(thresholds, recalls, label='Recall (Повнота)', color='green', linewidth=2)
# Лінія F1
plt.plot(thresholds, f1_scores, label='F1 Score (Баланс)', color='purple', linestyle='--', alpha=0.7)

# Позначаємо поточний поріг (0.467)
plt.axvline(x=0.467, color='red', linestyle=':', label='Твій поточний поріг (0.467)')

# Позначаємо найкращий поріг по F1
plt.axvline(x=best_f1_threshold, color='purple', linestyle=':', 
            label=f'Макс. F1 поріг ({best_f1_threshold:.3f})')

plt.title('Precision-Recall Trade-off vs Classification Threshold')
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.legend(loc='lower left')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.show()