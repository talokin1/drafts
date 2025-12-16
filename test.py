from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Data dictionary content
rows = [
    ("Ідентифікація компанії", "IDENTIFYCODE", "ЄДРПОУ компанії"),
    ("Ідентифікація компанії", "FULL_FIRM_NAME", "Повна офіційна назва компанії відповідно до державного реєстру"),
    ("Ідентифікація компанії", "OPF", "Організаційно-правова форма компанії (ТОВ, ПП, АТ тощо)"),
    ("Ідентифікація компанії", "FIRM_NAME", "Скорочена або комерційна назва компанії"),

    ("Реєстраційні та статусні дані", "ubki_actual_date", "Дата актуальності даних у реєстрі UBKI"),
    ("Реєстраційні та статусні дані", "registry_last_change_date", "Дата останньої зафіксованої зміни в державному реєстрі"),
    ("Реєстраційні та статусні дані", "REGISTRATION_DATE", "Дата первинної державної реєстрації юридичної особи"),
    ("Реєстраційні та статусні дані", "STATUS", "Поточний юридичний статус компанії"),

    ("Уповноважені особи", "AUTHORISED_NAME_N", "ПІБ уповноваженої особи компанії"),
    ("Уповноважені особи", "AUTHORISED_ROLE_N", "Роль відповідної уповноваженої особи"),

    ("Засновники", "FOUNDER_NAME_N", "ПІБ фізичної особи або назва юридичної особи — засновника компанії"),

    ("Бенефіціарні власники", "BENEFICIARY_NAME_N", "ПІБ кінцевого бенефіціарного власника"),
    ("Бенефіціарні власники", "BENEFICIARY_SHARE_N", "Частка володіння відповідного бенефіціара у відсотках (%)"),

    ("Види економічної діяльності (КВЕД)", "KVED", "Основний КВЕД компанії"),
    ("Види економічної діяльності (КВЕД)", "KVED_DESCR", "Опис основного виду економічної діяльності"),
    ("Види економічної діяльності (КВЕД)", "KVED_N", "Додаткові КВЕДи компанії"),
    ("Види економічної діяльності (КВЕД)", "KVED_N_DESCR", "Опис відповідних додаткових КВЕДів"),

    ("МСБ скоринг (UBKI)", "MSB_SCORE", "Числове значення МСБ-скорингу компанії"),
    ("МСБ скоринг (UBKI)", "MSB_LEVEL", "Категоріальний рівень МСБ-скорингу"),
    ("МСБ скоринг (UBKI)", "MSB_SCORE_DATE", "Дата розрахунку МСБ-скорингу"),
]

# Create workbook
wb = Workbook()
ws = wb.active
ws.title = "Data Dictionary"

# Styles
header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="2F5597")
center = Alignment(vertical="center", wrap_text=True)
border = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin")
)

# Header
headers = ["Section", "Field name", "Description"]
ws.append(headers)
for col in range(1, 4):
    cell = ws.cell(row=1, column=col)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = center
    cell.border = border

# Rows
for r in rows:
    ws.append(r)

for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=3):
    for cell in row:
        cell.alignment = center
        cell.border = border

# Auto width
for col in range(1, 4):
    ws.column_dimensions[get_column_letter(col)].width = [28, 30, 80][col-1]

# Freeze header
ws.freeze_panes = "A2"

# Save file
path = "/mnt/data/UBKI_Data_Dictionary.xlsx"
wb.save(path)

path
