# --- 8. ВПОРЯДКУВАННЯ КОЛОНОК (Reorder) ---
target_col = 'KVED_5_DESCR'
cols_to_move = ['OPF_CODE', 'OPF_NAME']

# Перевіряємо, чи існують всі необхідні колонки
if target_col in final_df.columns and all(c in final_df.columns for c in cols_to_move):
    # 1. Створюємо список колонок БЕЗ тих, що ми хочемо перемістити
    remaining_cols = [c for c in final_df.columns if c not in cols_to_move]
    
    # 2. Знаходимо індекс цільової колонки
    target_idx = remaining_cols.index(target_col)
    
    # 3. Формуємо новий порядок: [все до цілі] + [ціль] + [наші колонки] + [все інше]
    new_order = remaining_cols[:target_idx + 1] + cols_to_move + remaining_cols[target_idx + 1:]
    
    # 4. Застосовуємо новий порядок
    final_df = final_df[new_order]
    print(f"Колонки успішно переміщено за '{target_col}'.")
else:
    print(f"Увага: Колонку '{target_col}' не знайдено, порядок залишено без змін.")

# Перевірка
print(final_df.columns.tolist()[final_df.columns.get_loc('KVED_5_DESCR'):final_df.columns.get_loc('KVED_5_DESCR')+5])