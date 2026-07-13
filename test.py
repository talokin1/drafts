import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# Колонка з ID клієнта
CLIENT_COL = "CONTRAGENTAID"

# Один клієнт = еквайринг, якщо хоча б один його рядок має True
client_summary = (
    result
    .groupby(CLIENT_COL, as_index=False)
    .agg(is_acquiring=("is_acquiring", "max"))
)

summary = (
    client_summary["is_acquiring"]
    .map({
        True: "З еквайрингом",
        False: "Без еквайрингу"
    })
    .value_counts()
    .rename_axis("Категорія")
    .reset_index(name="Кількість клієнтів")
)

summary["Частка, %"] = (
    summary["Кількість клієнтів"]
    / summary["Кількість клієнтів"].sum()
    * 100
).round(2)

output_path = r"C:\Projects\(DS-398) Acquiring\acquiring_result.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    result.to_excel(writer, sheet_name="Data", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)

    ws = writer.book["Summary"]

    # Форматування заголовка
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin")
    )

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    # Форматування таблиці
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center")

    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 15
    ws.freeze_panes = "A2"

print(f"Файл збережено: {output_path}")