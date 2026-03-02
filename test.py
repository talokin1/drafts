import matplotlib.pyplot as plt
import seaborn as sns

# Витягуємо важливості типу 'gain'
importance_gain = clf_model.booster_.feature_importance(importance_type='gain')
feature_names = X_train.columns

# Створюємо DataFrame для зручності
df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_gain
})

# Нормалізуємо у відсотки (опціонально, але зручно для бізнесу)
df_importance['Importance_%'] = 100 * (df_importance['Importance'] / df_importance['Importance'].sum())

# Сортуємо від найважливіших до найменш важливих
df_importance = df_importance.sort_values(by='Importance_%', ascending=False).reset_index(drop=True)

# Виводимо топ-15 фічей у консоль
print("Топ-15 найважливіших фічей (за Gain):")
print(df_importance.head(15))

# Будуємо графік
plt.figure(figsize=(10, 8))
# Беремо тільки Топ-20 для графіка, щоб не перевантажувати візуал
sns.barplot(
    data=df_importance.head(20), 
    x='Importance_%', 
    y='Feature', 
    palette='viridis'
)
plt.title('Feature Importance (Gain) - Топ 20 ознак')
plt.xlabel('Внесок у зниження помилки (%)')
plt.ylabel('Ознака')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()