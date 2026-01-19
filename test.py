# Відбираємо тільки тих, у кого більше 0, але менше 100 грн (можна змінити на 200)
small_positive = df[(df['CURR_ACC'] > 0) & (df['CURR_ACC'] < 100)]['CURR_ACC']

print(f"Кількість клієнтів у зоні 0-100 грн: {len(small_positive)}")
print(small_positive.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.hist(small_positive, bins=50, edgecolor='black')
plt.title("Розподіл залишків від 0 до 100 грн")
plt.xlabel("Сума на рахунку")
plt.ylabel("Кількість клієнтів")
plt.show()