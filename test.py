import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Витягуємо важливість фічей
feature_imp = pd.DataFrame(sorted(zip(reg.feature_importances_, X.columns)), columns=['Value','Feature'])

# 2. Малюємо Топ-30
plt.figure(figsize=(10, 8))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:30])
plt.title('Top 30 Most Important Features')
plt.tight_layout()
plt.show()

# 3. Виводимо список топ-фічей текстом
top_features = feature_imp.sort_values(by="Value", ascending=False).head(20)['Feature'].tolist()
print("TOP 20 Features:", top_features)

# 4. Перевіряємо, скільки фічей взагалі не використовуються (важливість = 0)
zero_imp_features = feature_imp[feature_imp['Value'] == 0]['Feature'].tolist()
print(f"\nUseless features count: {len(zero_imp_features)}")