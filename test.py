from openpyxl.utils import get_column_letter # Додай це до імпортів на початку файлу

# Фінальне шліфування
for sheet in wb.sheetnames:
    ws = wb[sheet]
    # Використовуємо enumerate(..., 1), щоб мати точний числовий індекс колонки
    for i, col in enumerate(ws.columns, 1):
        max_length = 0
        column_letter = get_column_letter(i) # Безпечно отримуємо букву (A, B, C...)
        
        for cell in col:
            try:
                # Перевіряємо, чи комірка не порожня, перш ніж рахувати довжину
                if cell.value: 
                    cell_length = len(str(cell.value))
                    if cell_length > max_length:
                        max_length = cell_length
            except: 
                pass
        
        # Задаємо ширину з невеликим запасом
        ws.column_dimensions[column_letter].width = max_length + 3

wb.save("hnwi_potential_report_v1.xlsx")