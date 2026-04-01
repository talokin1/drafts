import pandas as pd
import numpy as np

def generate_business_insights(df, output_filename="Business_Insights_Dashboard.xlsx"):
    df_base = df.copy()
    
    # Створюємо маркер існуючого клієнта
    df_base['Client_Status'] = np.where(df_base['CONTRAGENTID'].notna(), 'Існуючий клієнт', 'Новий/Потенційний')

    # --- ТАБЛИЦЯ 1: Топ-20 КВЕДів ---
    kved_pivot = df_base.groupby('KVED_DESCR').agg(
        Count=('IDENTIFYCODE', 'count'),
        Total_Potential=('POTENTIAL_INCOME', 'sum')
    ).sort_values(by='Total_Potential', ascending=False).head(20)
    kved_pivot.columns = ['Кількість компаній', 'Сумарний потенціал (₴)']

    # --- ТАБЛИЦЯ 2: Аналітика за сегментами (FIRM_TYPE) ---
    segment_pivot = df_base.groupby('FIRM_TYPE').agg(
        Count=('IDENTIFYCODE', 'count'),
        Total_Potential=('POTENTIAL_INCOME', 'sum'),
        Avg_Potential=('POTENTIAL_INCOME', 'mean')
    ).sort_values(by='Total_Potential', ascending=False)
    segment_pivot.columns = ['Кількість', 'Загальний потенціал (₴)', 'Середній чек (₴)']

    # --- ТАБЛИЦЯ 3: Структура доходу за продуктами ---
    # Допомагає зрозуміти, які продукти драйвлять дохід у кожному сегменті
    product_cols = ['FX_POTENTIAL', 'TRANSACTION_POTENTIAL', 'ACCOUNTS_POTENTIAL', 'ASSETS_POTENTIAL', 'LIABILITIES_POTENTIAL']
    product_pivot = df_base.groupby('FIRM_TYPE')[product_cols].sum()
    product_pivot.columns = ['FX', 'Транзакції', 'Рахунки', 'Активи (Кредити)', 'Пасиви (Депозити)']
    # Сортуємо по загальному потенціалу (щоб порядок збігався з Таблицею 2)
    product_pivot = product_pivot.loc[segment_pivot.index]

    # --- ТАБЛИЦЯ 4: Існуючі клієнти проти Нових ---
    status_pivot = df_base.groupby('Client_Status').agg(
        Count=('IDENTIFYCODE', 'count'),
        Total_Potential=('POTENTIAL_INCOME', 'sum')
    ).sort_values(by='Total_Potential', ascending=False)
    status_pivot.columns = ['Кількість компаній', 'Потенціал до залучення (₴)']

    # ==========================================
    # ЗАПИС ТА ФОРМАТУВАННЯ В EXCEL
    # ==========================================
    writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
    workbook = writer.book
    worksheet = workbook.add_worksheet('Insights_Dashboard')
    
    # Вимикаємо сітку для вигляду дашборду
    worksheet.hide_gridlines(2)

    # Формати
    title_fmt = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#1F497D'})
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#1F497D', 'font_color': 'white', 'border': 1, 'align': 'center'})
    index_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
    money_fmt = workbook.add_format({'num_format': '#,##0 ₴', 'border': 1})
    count_fmt = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'center'})

    def write_pivot_to_excel(pivot_df, start_row, start_col, title):
        """Допоміжна функція для акуратного запису таблиць з форматуванням"""
        # Заголовок таблиці
        worksheet.write(start_row, start_col, title, title_fmt)
        
        # Заголовки колонок (включаючи назву індексу)
        worksheet.write(start_row + 1, start_col, pivot_df.index.name if pivot_df.index.name else "Категорія", header_fmt)
        for col_num, col_name in enumerate(pivot_df.columns):
            worksheet.write(start_row + 1, start_col + col_num + 1, col_name, header_fmt)
            
        # Запис даних
        for row_num, (index_val, row_data) in enumerate(pivot_df.iterrows()):
            worksheet.write(start_row + row_num + 2, start_col, str(index_val), index_fmt)
            
            for col_num, val in enumerate(row_data):
                # Визначаємо формат залежно від назви колонки (гроші чи кількість)
                col_name = pivot_df.columns[col_num]
                fmt = money_fmt if '₴' in col_name or 'FX' in col_name or 'Транзакції' in col_name or 'Рахунки' in col_name or 'Активи' in col_name or 'Пасиви' in col_name else count_fmt
                
                # Захист від NaN
                if pd.isna(val) or np.isinf(val):
                    val = 0
                worksheet.write_number(start_row + row_num + 2, start_col + col_num + 1, val, fmt)

    # Розміщення таблиць на аркуші (координати: рядок, колонка)
    # Зліва: КВЕДи
    write_pivot_to_excel(kved_pivot, start_row=1, start_col=1, title="1. Топ індустрій за потенціалом (КВЕД)")
    
    # Справа: Аналітика по сегментах та статусу
    write_pivot_to_excel(segment_pivot, start_row=1, start_col=5, title="2. Аналітика за розміром бізнесу")
    
    # Відступаємо вниз під Таблицею 2
    write_pivot_to_excel(product_pivot, start_row=len(segment_pivot) + 5, start_col=5, title="3. Структура потенційного доходу (Банківські продукти)")
    
    # Відступаємо вниз під Таблицею 3
    write_pivot_to_excel(status_pivot, start_row=len(segment_pivot) + len(product_pivot) + 9, start_col=5, title="4. Потенціал: Внутрішня база vs Ринок")

    # Налаштування ширини колонок для красивого відображення
    worksheet.set_column('A:A', 2)   # Відступ зліва
    worksheet.set_column('B:B', 50)  # Назви КВЕДів (широка)
    worksheet.set_column('C:C', 18)  # К-сть компаній (КВЕД)
    worksheet.set_column('D:D', 25)  # Гроші (КВЕД)
    worksheet.set_column('E:E', 5)   # Роздільник між блоками
    worksheet.set_column('F:F', 20)  # Назви сегментів/статусів
    worksheet.set_column('G:K', 18)  # Колонки з грошима справа

    writer.close()
    print(f"Дашборд {output_filename} успішно згенеровано!")

# Виклик:
# generate_business_insights(df_)