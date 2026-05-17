import pandas as pd
import numpy as np

# =========================
# 1. Налаштування
# =========================

output_path = "final_clients_analysis.xlsx"

# Копія, щоб не змінювати original final
df = final.copy()

# =========================
# 2. Логічний порядок колонок
# =========================

logical_order = [
    # Основна інформація про підприємство
    "IDENTIFYCODE",
    "FIRM_NAME",
    "FIRM_TERR",
    "KVED",
    "KVED_DESCR",

    # Фінансові показники
    "REVENUE_CUR",
    "NET_PROFIT_CUR",
    "NB_EMPL",
    "ФОП",

    # Банківська інформація
    "BANK_USED",

    # Контактна інформація
    "FIRM_RUK",
    "FIRM_TELORG",
]

# Беремо тільки ті колонки, які реально є в таблиці
logical_order = [col for col in logical_order if col in df.columns]

df = df[logical_order]

# =========================
# 3. Перейменування колонок
# =========================

rename_dict = {
    "IDENTIFYCODE": "ЄДРПОУ",
    "FIRM_NAME": "Назва підприємства",
    "FIRM_TERR": "Регіон",
    "KVED": "КВЕД",
    "KVED_DESCR": "Опис КВЕД",
    "REVENUE_CUR": "Дохід, тис. грн",
    "NET_PROFIT_CUR": "Чистий прибуток, тис. грн",
    "NB_EMPL": "Кількість працівників",
    "ФОП": "ФОП, тис. грн",
    "BANK_USED": "Банк",
    "FIRM_RUK": "Керівник",
    "FIRM_TELORG": "Телефон",
}

df = df.rename(columns=rename_dict)

# =========================
# 4. Базова очистка
# =========================

# Порожні банки замінюємо на "Немає інформації"
if "Банк" in df.columns:
    df["Банк"] = df["Банк"].fillna("Немає інформації")
    df["Банк"] = df["Банк"].replace("", "Немає інформації")

if "КВЕД" in df.columns:
    df["КВЕД"] = df["КВЕД"].astype(str).replace("nan", "Немає інформації")

if "Опис КВЕД" in df.columns:
    df["Опис КВЕД"] = df["Опис КВЕД"].fillna("Немає інформації")

# Числові колонки
num_cols = [
    "Дохід, тис. грн",
    "Чистий прибуток, тис. грн",
    "Кількість працівників",
    "ФОП, тис. грн",
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 5. Зведення по банках
# =========================

bank_pivot = (
    df
    .groupby("Банк", dropna=False)
    .agg(
        Кількість_підприємств=("ЄДРПОУ", "count"),
        Загальний_дохід_тис_грн=("Дохід, тис. грн", "sum"),
        Середній_дохід_тис_грн=("Дохід, тис. грн", "mean"),
        Загальний_чистий_прибуток_тис_грн=("Чистий прибуток, тис. грн", "sum"),
        Середній_чистий_прибуток_тис_грн=("Чистий прибуток, тис. грн", "mean"),
        Загальна_кількість_працівників=("Кількість працівників", "sum"),
        Загальний_ФОП_тис_грн=("ФОП, тис. грн", "sum"),
        Кількість_унікальних_КВЕДів=("КВЕД", "nunique"),
    )
    .reset_index()
    .sort_values("Кількість_підприємств", ascending=False)
)

# Частка підприємств по банку
bank_pivot["Частка_підприємств"] = (
    bank_pivot["Кількість_підприємств"] / bank_pivot["Кількість_підприємств"].sum()
)

# =========================
# 6. Зведення по КВЕДах
# =========================

kved_pivot = (
    df
    .groupby(["КВЕД", "Опис КВЕД"], dropna=False)
    .agg(
        Кількість_підприємств=("ЄДРПОУ", "count"),
        Загальний_дохід_тис_грн=("Дохід, тис. грн", "sum"),
        Середній_дохід_тис_грн=("Дохід, тис. грн", "mean"),
        Загальний_чистий_прибуток_тис_грн=("Чистий прибуток, тис. грн", "sum"),
        Середній_чистий_прибуток_тис_грн=("Чистий прибуток, тис. грн", "mean"),
        Загальна_кількість_працівників=("Кількість працівників", "sum"),
        Загальний_ФОП_тис_грн=("ФОП, тис. грн", "sum"),
        Кількість_унікальних_банків=("Банк", "nunique"),
    )
    .reset_index()
    .sort_values("Кількість_підприємств", ascending=False)
)

kved_pivot["Частка_підприємств"] = (
    kved_pivot["Кількість_підприємств"] / kved_pivot["Кількість_підприємств"].sum()
)

# =========================
# 7. Перетин Банк × КВЕД
# =========================

bank_kved_pivot = (
    df
    .groupby(["Банк", "КВЕД", "Опис КВЕД"], dropna=False)
    .agg(
        Кількість_підприємств=("ЄДРПОУ", "count"),
        Загальний_дохід_тис_грн=("Дохід, тис. грн", "sum"),
        Загальний_чистий_прибуток_тис_грн=("Чистий прибуток, тис. грн", "sum"),
        Загальна_кількість_працівників=("Кількість працівників", "sum"),
        Загальний_ФОП_тис_грн=("ФОП, тис. грн", "sum"),
    )
    .reset_index()
    .sort_values(["Банк", "Кількість_підприємств"], ascending=[True, False])
)

# =========================
# 8. Запис в Excel
# =========================

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    workbook = writer.book

    # Основний лист
    df.to_excel(writer, sheet_name="Дані", index=False)

    # Зведення
    sheet_name = "Зведення"
    start_row = 0

    worksheet_summary = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet_summary

    # Формати
    title_format = workbook.add_format({
        "bold": True,
        "font_size": 14,
        "bg_color": "#D9EAF7",
        "border": 1
    })

    header_format = workbook.add_format({
        "bold": True,
        "bg_color": "#BDD7EE",
        "border": 1,
        "text_wrap": True
    })

    number_format = workbook.add_format({
        "num_format": "#,##0.0"
    })

    integer_format = workbook.add_format({
        "num_format": "#,##0"
    })

    percent_format = workbook.add_format({
        "num_format": "0.0%"
    })

    # --- Таблиця 1: по банках
    worksheet_summary.write(start_row, 0, "Розподіл підприємств по банках", title_format)
    start_row += 2

    bank_pivot.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row,
        startcol=0,
        index=False
    )

    start_row += len(bank_pivot) + 4

    # --- Таблиця 2: по КВЕДах
    worksheet_summary.write(start_row, 0, "Розподіл підприємств по КВЕДах", title_format)
    start_row += 2

    kved_pivot.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row,
        startcol=0,
        index=False
    )

    start_row += len(kved_pivot) + 4

    # --- Таблиця 3: Банк × КВЕД
    worksheet_summary.write(start_row, 0, "Розподіл підприємств за перетином Банк × КВЕД", title_format)
    start_row += 2

    bank_kved_pivot.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row,
        startcol=0,
        index=False
    )

    # =========================
    # 9. Форматування листа "Дані"
    # =========================

    worksheet_data = writer.sheets["Дані"]

    worksheet_data.freeze_panes(1, 0)
    worksheet_data.autofilter(0, 0, len(df), len(df.columns) - 1)

    for col_num, col_name in enumerate(df.columns):
        worksheet_data.write(0, col_num, col_name, header_format)

        max_len = max(
            df[col_name].astype(str).map(len).max(),
            len(col_name)
        )

        width = min(max_len + 2, 45)
        worksheet_data.set_column(col_num, col_num, width)

    # Формати для числових колонок на листі "Дані"
    for col_name in num_cols:
        if col_name in df.columns:
            col_idx = df.columns.get_loc(col_name)

            if col_name == "Кількість працівників":
                worksheet_data.set_column(col_idx, col_idx, 18, integer_format)
            else:
                worksheet_data.set_column(col_idx, col_idx, 20, number_format)

    # =========================
    # 10. Форматування листа "Зведення"
    # =========================

    worksheet_summary.freeze_panes(0, 0)

    # Ширини колонок
    worksheet_summary.set_column(0, 0, 28)
    worksheet_summary.set_column(1, 1, 45)
    worksheet_summary.set_column(2, 20, 20)

    # Форматування заголовків таблиць на зведенні
    for row in range(0, start_row + len(bank_kved_pivot) + 10):
        # Проходимо тільки по потенційних header rows
        pass

    # Автоформатування заголовків через пошук рядків із назвами колонок
    for table_start in [
        2,
        2 + len(bank_pivot) + 4,
        2 + len(bank_pivot) + 4 + len(kved_pivot) + 4,
    ]:
        for col_num in range(0, 15):
            worksheet_summary.write(
                table_start,
                col_num,
                worksheet_summary.table.get if False else ""
            )

    # Простий варіант: повторно записуємо header-формат для кожної таблиці
    bank_header_row = 2
    kved_header_row = bank_header_row + len(bank_pivot) + 4
    bank_kved_header_row = kved_header_row + len(kved_pivot) + 4

    for col_num, col_name in enumerate(bank_pivot.columns):
        worksheet_summary.write(bank_header_row, col_num, col_name, header_format)

    for col_num, col_name in enumerate(kved_pivot.columns):
        worksheet_summary.write(kved_header_row, col_num, col_name, header_format)

    for col_num, col_name in enumerate(bank_kved_pivot.columns):
        worksheet_summary.write(bank_kved_header_row, col_num, col_name, header_format)

    # Формат відсотків
    if "Частка_підприємств" in bank_pivot.columns:
        col_idx = bank_pivot.columns.get_loc("Частка_підприємств")
        worksheet_summary.set_column(col_idx, col_idx, 18, percent_format)

    # Зберігається автоматично після виходу з with

print(f"Excel-файл збережено: {output_path}")