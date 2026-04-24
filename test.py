# --- 8. АГРЕГОВАНІ ПОКАЗНИКИ (Для звітності) ---

# 1. Показники по клієнтах
total_clients_count = len(clients_ids)
wo_taxes_clients_count = len(final_export_df)

# Розрахунок частки (Share = Part / Total)
wo_taxes_clients_share = wo_taxes_clients_count / total_clients_count

# 2. Показники по "відомостях" (транзакціях / записах)
# Загальна кількість транзакцій наших клієнтів за півроку
total_trx_count = len(client_trx)

# Кількість транзакцій, які Є податками (ПДФО, ЄСВ, ВЗ)
tax_trx_count = len(tax_transactions)

# Розрахунок частки податкових відомостей у загальному пулі
tax_trx_share = tax_trx_count / total_trx_count if total_trx_count > 0 else 0

# 3. Розподіл клієнтів без податків за їхнім бізнес-сегментом
segment_agg = final_export_df.groupby('Сегмент клієнта').size().reset_index(name='Кількість')
segment_agg['Частка, %'] = (segment_agg['Кількість'] / wo_taxes_clients_count * 100).round(2)


# --- ВИВІД РЕЗУЛЬТАТІВ ---

print("=== 2. Агреговані показники ===")
print(f"Загальна база клієнтів (унікальних ЄДРПОУ): {total_clients_count}")
print("-" * 40)
print(f"Кількість клієнтів без податків: {wo_taxes_clients_count}")
print(f"Частка таких клієнтів:           {wo_taxes_clients_share:.1%} ({wo_taxes_clients_share * 100:.2f}%)")
print("-" * 40)
print(f"Загальна кількість транзакцій клієнтів: {total_trx_count}")
print(f"Кількість податкових відомостей:        {tax_trx_count}")
print(f"Частка податкових відомостей:           {tax_trx_share:.1%} ({tax_trx_share * 100:.2f}%)")
print("=" * 40)

print("\nДеталізація цільових клієнтів за сегментами:")
display(segment_agg)