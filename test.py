import matplotlib.pyplot as plt
import seaborn as sns

# 1. Візуальний пошук (дивимося тільки на суми до 50 000)
plt.figure(figsize=(10, 6))
# Беремо тільки тих, у кого < 50k, і розбиваємо на 100 кошиків
sns.histplot(df[df["CURR_ACC"] < 50000]["CURR_ACC"], bins=100)
plt.title("Розподіл залишків до 50 000")
plt.show()

# 2. Математичний пошук (квантилі)
# Це покаже, який відсоток людей має суму нижче певного значення
print(df["CURR_ACC"].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))