import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
# Використовуємо ваш DataFrame з попереднього кроку
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(30))
plt.title('Top 30 Most Important Features')
plt.tight_layout()

# Збереження картинки
plt.savefig('feature_importance_plot.png', dpi=300)
plt.show()