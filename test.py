# 1. Створюємо основний датафрейм з результатами
# Припускаємо, що X_val зберіг індекси оригінального df, де є IDENTIFYCODE
report_df = pd.DataFrame(index=x_val.index)

# 2. Додаємо ідентифікатор (якщо він у колонках X_val або в окремому df)
# Якщо IDENTIFYCODE є в оригінальному df_raw:
report_df['IDENTIFYCODE'] = df_raw.loc[x_val.index, 'IDENTIFYCODE']

# 3. Додаємо реальні гроші та прогнози (з твого коду)
report_df['True_Money'] = np.expm1(y_val_log)
report_df['Pred_Original'] = np.expm1(y_pred_log)
report_df['Pred_Potential'] = np.expm1(y_pred_stretched)

# 4. Додаємо дециль для розуміння рангу клієнта
report_df['Decile'] = pd.qcut(report_df['Pred_Potential'], 10, labels=False) + 1

# 5. Розраховуємо помилку в грошовому еквіваленті
report_df['Abs_Error'] = (report_df['True_Money'] - report_df['Pred_Potential']).abs()

# Сортуємо за потенціалом (від найбільшого до найменшого)
report_df = report_df.sort_values('Pred_Potential', ascending=False)