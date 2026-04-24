# --- РОЗРАХУНОК АГРЕГОВАНИХ ПОКАЗНИКІВ (Пункт 2 з ТЗ) ---

# 1. Показники по клієнтах
total_clients = len(clients_ids)                       # Всі клієнти з файлів df1 та df2
wo_taxes_clients_count = len(final_export_df)          # Ті самі 601 клієнт

# Розрахунок частки
wo_taxes_clients_share = wo_taxes_clients_count / total_clients if total_clients > 0 else 0


# 2. Розподіл цих клієнтів за сегментами
# Групуємо відфільтрованих клієнтів (final_export_df) по колонці 'Сегмент клієнта'
segment_breakdown = final_export_df.groupby('Сегмент клієнта').size().reset_index(name='Кількість клієнтів')
segment_breakdown['Частка в сегменті, %'] = (segment_breakdown['Кількість клієнтів'] / wo_taxes_clients_count * 100).round(2)


# 3. Показники по "відповідних відомостях" (транзакціях)
total_trx = len(client_trx)                            # Всі транзакції клієнтів за період
tax_trx_count = len(tax_transactions)                  # Транзакції, що класифіковані як ПДФО, ЄСВ або ВЗ

# Розрахунок частки податкових транзакцій
tax_trx_share = tax_trx_count / total_trx if total_trx > 0 else 0


# --- ВИВІД ДЛЯ ВІДПРАВКИ КОЛЕГАМ ---

print("=== 2. Агреговані показники ===")
print(f"Загальна кількість досліджуваних клієнтів: {total_clients}")
print(f"Кількість клієнтів, що НЕ платять податки: {wo_taxes_clients_count}")
print(f"Частка таких клієнтів: {wo_taxes_clients_share:.1%} ({wo_taxes_clients_share * 100:.2f}%)\n")

print("Розподіл цих клієнтів за сегментами:")
display(segment_breakdown)

print("\n--- Аналітика по відомостях (транзакціях) ---")
print(f"Загальна кількість знайдених транзакцій за період: {total_trx}")
print(f"Кількість відповідних відомостей (сплата податків): {tax_trx_count}")
print(f"Частка податкових відомостей: {tax_trx_share:.1%} ({tax_trx_share * 100:.2f}%)")