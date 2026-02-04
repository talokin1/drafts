import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# --- НАЛАШТУВАННЯ ШЛЯХІВ ---
# Створіть папку 'models', якщо її немає, щоб не отримати помилку
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)

MODEL_NAME = 'curr_acc_quantile_lgbm'

# --- 1. ЗБЕРЕЖЕННЯ МОДЕЛІ ---
print(f"🔄 Зберігаю модель '{MODEL_NAME}'...")

# A. Зберігаємо як joblib (зручно для Python: зберігає параметри, sklearn-обгортку)
joblib.dump(reg, f'{save_dir}/{MODEL_NAME}.pkl')

# B. Зберігаємо як txt (стабільний архів структури дерев LightGBM)
reg.booster_.save_model(f'{save_dir}/{MODEL_NAME}.txt')

print("✅ Модель успішно збережена (.pkl та .txt)!")


# --- 2. FEATURE IMPORTANCE (FI) ---
print("\n📊 Рахую важливість ознак...")

# Отримуємо важливість (split - кількість використань у деревах)
# Використовуємо feature_name() з бустера, щоб гарантувати правильний порядок назв
fi_df = pd.DataFrame({
    'Feature': reg.booster_.feature_name(),
    'Importance': reg.booster_.feature_importance(importance_type='split')
})

# Сортуємо: від найважливіших до найменш важливих
fi_df = fi_df.sort_values(by='Importance', ascending=False)

# А. Зберігаємо таблицю в CSV (для Excel/звітів)
csv_path = f'{save_dir}/{MODEL_NAME}_feature_importance.csv'
fi_df.to_csv(csv_path, index=False)
print(f"✅ Таблиця FI збережена: {csv_path}")

# Б. Малюємо та зберігаємо графік
plt.figure(figsize=(12, 10))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=fi_df.head(30),  # Топ-30
    palette='viridis'
)
plt.title(f'Top 30 Features: {MODEL_NAME} (Split Importance)')
plt.xlabel('Importance (Times used in split)')
plt.tight_layout()

# Зберігаємо картинку
plot_path = f'{save_dir}/{MODEL_NAME}_fi_plot.png'
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"✅ Графік FI збережено: {plot_path}")


# --- 3. АНАЛІЗ "СМІТТЯ" (Zero Importance) ---
zero_imp_features = fi_df[fi_df['Importance'] == 0]['Feature'].tolist()

print(f"\n🗑️ Знайдено ознак з нульовою важливістю: {len(zero_imp_features)}")
if len(zero_imp_features) > 0:
    print("Приклад перших 5 сміттєвих фічей:", zero_imp_features[:5])
    
    # Збережемо список сміттєвих фічей, щоб виключити їх наступного разу
    with open(f'{save_dir}/useless_features.txt', 'w') as f:
        for item in zero_imp_features:
            f.write(f"{item}\n")
    print("✅ Список непотрібних фічей збережено у 'models/useless_features.txt'")