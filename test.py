import matplotlib.pyplot as plt
import seaborn as sns

# 1. Створюємо датафрейм з результатами
results = pd.DataFrame({
    'Actual': y_test_real,
    'Predicted': preds_real
})

# 2. Будуємо графік
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual', y='Predicted', data=results, alpha=0.5)

# 3. Додаємо ідеальну лінію (x=y)
max_val = max(results['Actual'].max(), results['Predicted'].max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')

plt.title(f'Actual vs Predicted (R2: {r2:.4f})')
plt.xlabel('Реальний залишок (Actual)')
plt.ylabel('Прогноз (Predicted)')
plt.xscale('log') # Логарифмічна шкала, бо у вас там великі розкиди
plt.yscale('log')
plt.grid(True)
plt.show()

# 4. Дивимось на "хвости" (найбільші помилки)
results['Error'] = results['Actual'] - results['Predicted']
results['AbsError'] = results['Error'].abs()
print("ТОП-5 найгірших прогнозів:")
print(results.sort_values('AbsError', ascending=False).head(5))